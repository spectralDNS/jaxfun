# Write ParaView snapshots from checkpoints that already exist.
#
# TurbulentChannel3D.py writes snapshots as it runs, when the case asks for it.
# This script covers the other case: a run that has already been done, whose
# checkpoints hold the state but nothing a visualization tool can read. It
# restores one or more checkpoints and writes the velocity to the same HDF5 +
# XDMF pair the solver would have written.
#
# With --stats it writes the running statistics instead, one .npz file per
# checkpoint, named after the step. Read one back with
#
#   stats = ChannelStatistics.load("statistics_2000.npz")   # profiles(), ...
#
# or, without TurbulentChannel3D.py, with np.load: the file holds the averaged
# profiles in wall units under their own keys ("y+", "U+", "urms+", ...) next to
# the raw sums the class reads back.
#
# With --spectrum it writes the wall-normal spectrum instead, one .npz file per
# checkpoint, plus a plot of it when matplotlib is available. Each of u, v, w and
# g = omega_z is mapped to the orthogonal basis (Chebyshev or Legendre) and
# reduced over the horizontal wavenumbers, leaving one number per wall-normal
# mode k: the largest |coefficient| over (kx, ky) under "<field>_max" and the
# root-sum-square under "<field>_rss". The (kx, ky) = (0, 0) mode is left out of
# both and reported on its own as "U_mean" and "V_mean". A well-resolved field
# decays by many orders of magnitude towards the last modes; a tail that flattens
# or turns up there is under-resolution or wall-normal aliasing, and it shows up
# first in the near-wall statistics, where the top basis functions are steepest.
#
# The case file has to be the one the checkpoints were written with: the grid,
# the box and the basis all have to match, and restore() says so if they do not.
#
# Usage:
#
#   python SnapshotFromCheckpoint.py runs/Re180/re180.toml            # latest
#   python SnapshotFromCheckpoint.py runs/Re180/re180.toml --step 2000
#   python SnapshotFromCheckpoint.py runs/Re180/re180.toml --all
#   python SnapshotFromCheckpoint.py case.toml -o /tmp/look.h5 --open
#   python SnapshotFromCheckpoint.py runs/Re180/re180.toml --all --stats
#   python SnapshotFromCheckpoint.py runs/Re180/re180.toml --spectrum
#
# Not a demo: it needs checkpoints that a previous run left behind.

# ruff: noqa: E402
import argparse
import contextlib
import os
import sys
from typing import Any

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]

import jax

jax.config.update("jax_enable_x64", True)

from spmd_bootstrap import echo, initialize_distributed, is_leader, to_host

initialize_distributed()

import jax.numpy as jnp
import numpy as np
from channel_case import load_case
from TurbulentChannel3D import (
    ChannelCheckpointer,
    ChannelStatistics,
    TurbulentChannel,
    build_solver,
    physical_velocity,
)

from jaxfun.typing import Array
from jaxfun.utils.hdf5file import HDF5File

SPECTRUM_FIELDS = ("u", "v", "w", "g")


def read_statistics(
    solver: TurbulentChannel, source: ChannelCheckpointer, step: int | None = None
) -> tuple[ChannelStatistics, int]:
    """Return the statistics saved with a checkpoint, and its step.

    Reads only the checkpoint's metadata, not the state. The statistics are
    sampled on the padded wall-normal mesh, so `solver` has to have the grid
    and Re_tau they were collected with.
    """
    meta = source.meta(step)
    for key, have in (
        ("grid", list(solver.grid)),
        ("Re_tau", float(solver.Re_tau)),
    ):
        if meta[key] != have:
            raise ValueError(
                f"checkpoint {key}={meta[key]!r} does not match this case's {have!r}"
            )
    z = to_host(solver.VD.mesh(N=solver.pad, broadcast=False)[2])
    stats = ChannelStatistics(z, float(solver.Re_tau))
    stats.load_dict(meta["stats"])
    return stats, int(meta["step"])


def write_statistics(
    solver: TurbulentChannel,
    source: ChannelCheckpointer,
    steps: list[int] | list[None],
    stem: str,
) -> None:
    """Write the statistics of each checkpoint in `steps` to `<stem>_<step>.npz`."""
    for step in steps:
        stats, step = read_statistics(solver, source, step)
        path = f"{stem}_{step}.npz"
        if is_leader():
            stats.save(path)
        span = (
            f" over t in [{stats.t_first:.3f}, {stats.t_last:.3f}]"
            if stats.count
            else ""
        )
        echo(f"  step {step}: {stats.count} samples{span} -> {path}")


@jax.jit
def wall_normal_spectrum(
    solver: TurbulentChannel, state: tuple[Array, ...]
) -> tuple[Array, Array, Array]:
    """Return the wall-normal spectrum of u, v, w and g, and of the mean flow.

    Returns the max and the root-sum-square over (kx, ky), each of shape
    (4, N) in `SPECTRUM_FIELDS` order with the (0, 0) mode excluded, and the
    orthogonal coefficients of the two mean profiles, shape (2, N).
    """
    w_hat, g_hat, u0, v0 = state[0], state[1], state[2], state[3]
    u_hat, v_hat = solver.velocity(w_hat, g_hat, u0, v0)
    cw = solver.VB.to_orthogonal(w_hat)
    cu, cv, cg = jax.vmap(solver.VD.to_orthogonal)(jnp.stack((u_hat, v_hat, g_hat)))
    a = jnp.abs(jnp.stack((cu, cv, cw, cg)).at[:, 0, 0].set(0))
    mean = jnp.abs(jax.vmap(solver.D1.to_orthogonal)(jnp.stack((u0, v0))))
    return a.max(axis=(1, 2)), jnp.sqrt((a**2).sum(axis=(1, 2))), mean


def plot_spectrum(spectrum: dict[str, np.ndarray], path: str, title: str) -> None:
    """Plot the spectrum `write_spectra` saved, on a log scale, to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    k = spectrum["k"]
    for ax, kind in zip(axes, ("max", "rss"), strict=True):
        for name in SPECTRUM_FIELDS:
            ax.semilogy(k, spectrum[f"{name}_{kind}"], label=name)
        ax.semilogy(k, spectrum["U_mean"], "k--", label="U (mean)")
        ax.set_xlabel("wall-normal mode k")
        ax.set_title(f"{kind} over (kx, ky)")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("|coefficient|")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_spectra(
    solver: TurbulentChannel,
    source: ChannelCheckpointer,
    steps: list[int] | list[None],
    stem: str,
    tail: int = 8,
) -> None:
    """Write the wall-normal spectrum of each checkpoint to `<stem>_<step>.npz`.

    Also prints, per field, how far the last `tail` modes lie below the peak.
    """
    z = to_host(solver.VD.mesh(N=solver.pad, broadcast=False)[2])
    stats = ChannelStatistics(z, float(solver.Re_tau))
    for step in steps:
        state, t, step, _ = source.restore(solver, stats, step)
        amax, rss, mean = to_host(wall_normal_spectrum(solver, state))
        spectrum: dict[str, Any] = {"k": np.arange(amax.shape[1]), "t": t}
        for i, name in enumerate(SPECTRUM_FIELDS):
            spectrum[f"{name}_max"] = amax[i]
            spectrum[f"{name}_rss"] = rss[i]
        spectrum["U_mean"], spectrum["V_mean"] = mean[0], mean[1]
        path = f"{stem}_{step}.npz"
        echo(f"  step {step} at t={t:.3f} -> {path}")
        echo(f"    {'field':>6} {'peak':>10} {'tail max':>10} {'tail/peak':>10}")
        for name in (*SPECTRUM_FIELDS, "U"):
            s = spectrum[f"{name}_max" if name != "U" else "U_mean"]
            peak, top = s.max(), s[-tail:].max()
            echo(f"    {name:>6} {peak:10.3e} {top:10.3e} {top / peak:10.3e}")
        if is_leader():
            np.savez(path, **spectrum)
            with contextlib.suppress(ImportError):
                plot_spectrum(spectrum, f"{stem}_{step}.png", f"step {step}, t={t:.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="The case file the checkpoints belong to.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoints to read; the case's checkpoint_dir by default.",
    )
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step.")
    parser.add_argument(
        "--all", action="store_true", help="Every checkpoint, not just one."
    )
    parser.add_argument(
        "-o",
        dest="out",
        default=None,
        help="Output HDF5 file; the case's snapshot_file by default.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Write the running statistics, one .npz per checkpoint, instead of "
        "snapshots; -o then gives the file name stem.",
    )
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help="Write the wall-normal spectrum of u, v, w and g, one .npz (and .png) "
        "per checkpoint, instead of snapshots; -o then gives the file name stem.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Store on the mesh as it is, leaving the walls out and the periodic "
        "ends open.",
    )
    args = parser.parse_args(argv)

    case = load_case(args.case)
    solver = build_solver(case)
    directory = args.checkpoint_dir or case.path("checkpoint_dir")
    source = ChannelCheckpointer(directory)
    steps = source.manager.all_steps() if args.all else [args.step]
    if not steps:
        raise FileNotFoundError(f"no checkpoint in {directory}")
    if args.stats:
        stem = os.path.splitext(args.out or case.base_dir / "statistics")[0]
        write_statistics(solver, source, sorted(steps), stem)
        source.close()
        return
    if args.spectrum:
        stem = os.path.splitext(args.out or case.base_dir / "spectrum")[0]
        write_spectra(solver, source, sorted(steps), stem)
        source.close()
        return

    mesh = to_host(solver.VD.mesh(broadcast=False))
    out = args.out or case.path("snapshot_file")
    snapshots = None
    if is_leader():
        snapshots = HDF5File.from_coords(
            out,
            mesh,
            domains=[(0.0, case.Lx), (0.0, case.Ly), (-1.0, 1.0)],
            dtype=np.float64 if case.snapshot_float64 else np.float32,
            wrap_axes=() if args.open else (0, 1),
            wall_axes=() if args.open else (2,),
        )

    z = to_host(solver.VD.mesh(N=solver.pad, broadcast=False)[2])
    stats = ChannelStatistics(z, case.Re_tau)
    for step in sorted(steps):
        state, t, step, _ = source.restore(solver, stats, step)
        u, v, w = to_host(physical_velocity(solver, state))
        if snapshots is not None:
            snapshots.write({"U": np.stack((u, v, w))}, time=t, step=step)
        echo(f"  step {step} at t={t:.3f}")
    source.close()
    if snapshots is not None:
        echo(f"  wrote {snapshots.xdmf_path}")
        snapshots.close()


if __name__ == "__main__":
    main()
