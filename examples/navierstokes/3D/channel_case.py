"""Case files for TurbulentChannel3D.py: one TOML file describes one run.

A case file sets any subset of the keys below; the rest keep their defaults,
which are the production Re_tau = 180 run. `cases/re180.toml` lists every key
with a comment and is the place to start a new case from::

    [flow]            Re_tau, Lx, Ly
    [grid]            Nx, Ny, Nz, Px, Py, Pz
    [discretization]  polynomial, kind, tableau
    [time]            dt, t_transient, t_end, regrid_transient
    [output]          sample_every, checkpoint_every, log_every,
                      checkpoint_dir, plot, snapshot_every, snapshot_file,
                      snapshot_closed, snapshot_float64
    [init]            seed

Relative output paths are taken relative to the case file, not the working
directory, so a case can live anywhere -- a sandbox, a scratch disk -- and its
checkpoints and figures land beside it while the solver stays where it is.

Not a demo: imported by TurbulentChannel3D.py and its benchmark.
"""

import argparse
import dataclasses
import math
import os
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SECTIONS: dict[str, tuple[str, ...]] = {
    "flow": ("Re_tau", "Lx", "Ly"),
    "grid": ("Nx", "Ny", "Nz", "Px", "Py", "Pz"),
    "discretization": ("polynomial", "kind", "tableau"),
    "time": ("dt", "t_transient", "t_end", "regrid_transient"),
    "output": (
        "sample_every",
        "checkpoint_every",
        "log_every",
        "checkpoint_dir",
        "plot",
        "snapshot_every",
        "snapshot_file",
        "snapshot_closed",
        "snapshot_float64",
    ),
    "init": ("seed",),
}


@dataclass(frozen=True)
class ChannelCase:
    """Everything that defines a turbulent channel run.

    Attributes:
        Re_tau: Friction Reynolds number.
        Lx: Box size (streamwise)
        Ly: Box size (spanwise)
        Nx: Fourier modes (streamwise)
        Ny: Fourier modes (spanwise)
        Nz: Polynomial modes (wall-normal)
        Px: Dealiasing factor (streamwise)
        Py: Dealiasing factor (spanwise)
        Pz: Dealiasing factor (wall-normal)
        polynomial: Wall-normal basis, a `PolynomialKind` name or value.
        kind: Test space, a `TestSpaceKind` name or value.
        tableau: Name of a globally stiffly accurate IMEX tableau in
            `jaxfun.integrators`.
        dt: Time step, in h/u_tau.
        t_transient: Time before statistics are collected.
        t_end: Time to integrate to.
        regrid_transient: Extra transient after a restart on a new grid,
            wall-normal padding or Re_tau.
        sample_every: Steps per chunk; statistics are sampled once per chunk.
        checkpoint_every: Chunks between checkpoints.
        log_every: Chunks between log lines.
        checkpoint_dir: Checkpoint directory.
        plot: File the profile figure is saved to.
        snapshot_every: Chunks between ParaView snapshots; 0 writes none.
        snapshot_file: HDF5 file the snapshots go to, with an XDMF sidecar
            written beside it.
        snapshot_closed: Store the snapshots on a mesh closed at the periodic
            ends and at the two walls, so ParaView draws no seam and no missing
            skin. Correct here because the velocity is zero on both walls.
        snapshot_float64: Store snapshots in float64 instead of float32.
        seed: Seed of the turbulent initial condition.
        base_dir: Directory relative output paths are resolved against: the
            case file's, or the working directory without one.
    """

    Re_tau: float = 180.0
    Lx: float = 2 * math.pi
    Ly: float = math.pi
    Nx: int = 64
    Ny: int = 64
    Nz: int = 64
    Px: float = 1.5
    Py: float = 1.5
    Pz: float = 1.0
    polynomial: str = "chebyshev"
    kind: str = "galerkin_recombined"
    tableau: str = "ARS443"
    dt: float = 0.001
    t_transient: float = 20.0
    t_end: float = 1.0
    regrid_transient: float = 5.0
    sample_every: int = 100
    checkpoint_every: int = 10
    log_every: int = 10
    checkpoint_dir: str = "turbulent_channel_ckpt"
    plot: str = "turbulent_channel_profiles.png"
    snapshot_every: int = 0
    snapshot_file: str = "snapshots.h5"
    snapshot_closed: bool = True
    snapshot_float64: bool = False
    seed: int = 1
    base_dir: Path = field(default_factory=Path.cwd)

    @property
    def padded(self) -> tuple[int, int, int]:
        """Shape of the physical mesh."""
        return (
            int(self.Px * self.Nx),
            int(self.Py * self.Ny),
            int(self.Pz * self.Nz),
        )

    def path(self, name: str) -> Path:
        """Return output path `name` resolved against `base_dir`."""
        return self.base_dir / getattr(self, name)


def pytest_case() -> ChannelCase:
    """Return the small, short case the test suite runs."""
    return ChannelCase(
        Nx=16,
        Ny=16,
        Nz=24,
        t_transient=0.01,
        t_end=0.02,
        sample_every=5,
        checkpoint_every=2,
        log_every=1,
    )


def _typed(name: str, value: Any) -> Any:
    """Return `value` as the type field `name` holds, or raise."""
    default = ChannelCase.__dataclass_fields__[name].default
    if (
        isinstance(default, float)
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    if type(value) is not type(default):
        raise TypeError(
            f"{name} = {value!r} should be a {type(default).__name__}, "
            f"not a {type(value).__name__}"
        )
    return value


def _flatten(table: dict[str, Any], source: str) -> dict[str, Any]:
    """Return the `[section] key = value` entries of `table` as field values."""
    values: dict[str, Any] = {}
    for section, entries in table.items():
        if section not in SECTIONS:
            raise KeyError(
                f"{source}: unknown section [{section}]; "
                f"valid sections are {', '.join(SECTIONS)}"
            )
        if not isinstance(entries, dict):
            raise TypeError(f"{source}: [{section}] must be a table")
        for key, value in entries.items():
            if key not in SECTIONS[section]:
                raise KeyError(
                    f"{source}: unknown key {section}.{key}; valid keys in "
                    f"[{section}] are {', '.join(SECTIONS[section])}"
                )
            values[key] = _typed(key, value)
    return values


def _parse_override(item: str) -> dict[str, Any]:
    """Return `section.key=value` as a one-entry nested table."""
    lhs, sep, rhs = item.partition("=")
    section, dot, key = lhs.strip().partition(".")
    if not (sep and dot):
        raise ValueError(f"--set {item!r}: expected section.key=value")
    try:
        value = tomllib.loads(f"v = {rhs.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = rhs.strip()  # a bare word, e.g. --set discretization.kind=galerkin
    return {section: {key: value}}


def load_case(
    path: str | os.PathLike | None, overrides: Sequence[str] = ()
) -> ChannelCase:
    """Read a case file and apply `section.key=value` overrides on top of it.

    Args:
        path: TOML case file, or None for the defaults.
        overrides: Values as they would be written in the file, e.g.
            `flow.Re_tau=185` or `discretization.kind="galerkin"`.
    """
    values: dict[str, Any] = {}
    base_dir = Path.cwd()
    if path is not None:
        path = Path(path).expanduser().resolve()
        with open(path, "rb") as f:
            values.update(_flatten(tomllib.load(f), str(path)))
        base_dir = path.parent
    for item in overrides:
        values.update(_flatten(_parse_override(item), "--set"))
    return ChannelCase(**values, base_dir=base_dir)


def to_dict(case: ChannelCase) -> dict[str, Any]:
    """Return `case` as a JSON-safe dict, for checkpoint metadata."""
    d = dataclasses.asdict(case)
    d["base_dir"] = str(case.base_dir)
    return d


def parse_cli(argv: list[str] | None = None) -> tuple[ChannelCase, argparse.Namespace]:
    """Parse TurbulentChannel3D.py's command line into a case and run options."""
    parser = argparse.ArgumentParser(
        description="Turbulent channel flow with the 3D KMM solver."
    )
    parser.add_argument("case", nargs="?", help="TOML case file (default: built-in)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one case-file entry; may be repeated.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument(
        "--restart-from",
        default=None,
        help="Start from the latest checkpoint in this directory instead of the "
        "checkpoint directory; its grid may differ from this run's.",
    )
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step.")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--average-from", type=float, default=None)
    parser.add_argument("--reset-stats", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    case = load_case(args.case, args.overrides)
    changes: dict[str, Any] = {}
    if args.t_end is not None:
        changes["t_end"] = args.t_end
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.checkpoint_dir is not None:
        # Given on the command line, so relative to where the command was run.
        changes["checkpoint_dir"] = os.path.abspath(args.checkpoint_dir)
    return dataclasses.replace(case, **changes), args
