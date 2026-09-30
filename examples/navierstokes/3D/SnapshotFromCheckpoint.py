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
#
# Not a demo: it needs checkpoints that a previous run left behind.

# ruff: noqa: E402
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]

import jax

jax.config.update("jax_enable_x64", True)

from spmd_bootstrap import echo, initialize_distributed, is_leader, to_host

initialize_distributed()

import numpy as np
from channel_case import load_case
from TurbulentChannel3D import (
    ChannelCheckpointer,
    ChannelStatistics,
    TurbulentChannel,
    build_solver,
    physical_velocity,
)

from jaxfun.utils.hdf5file import HDF5File


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
