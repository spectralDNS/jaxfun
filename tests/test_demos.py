import contextlib
import runpy
from pathlib import Path

import pytest

root = Path(__file__).parent.parent

# Modules that live in examples/ but are not demos: imported by a demo rather
# than run on their own. `spmd_bootstrap` brings up `jax.distributed` for the
# demos that can run under `mpirun`; running it on its own does nothing.
NOT_DEMOS = {"OrrSommerfeld_eigs", "ChannelFlow2D", "spmd_bootstrap"}

# Demos are grouped in subdirectories by topic, so this recurses. `notebooks/`
# is not part of the suite: it holds paired .py/.ipynb sources, several of which
# only make sense run cell by cell.
_all_files = [
    f
    for f in root.glob("examples/**/*.py")
    if f.is_file() and "notebooks" not in f.relative_to(root).parts
]
_all_files = [
    f for f in _all_files if "DrivenCavity" not in f.stem and f.stem not in NOT_DEMOS
]

# Parametrized on the stem alone, so a demo keeps its test id wherever it is
# filed and `-k <name>` goes on working. That only holds while stems are unique.
demo_paths: dict[str, Path] = {}
for f in _all_files:
    if f.stem in demo_paths:
        raise RuntimeError(
            f"Two demos are named {f.stem}.py ({demo_paths[f.stem]} and {f}); "
            "test ids are stems, so one would shadow the other."
        )
    demo_paths[f.stem] = f

files = [f.stem for f in _all_files]

# Demos that are run a second time under `--num-devices=2` (or 4). Nothing in a
# demo marks it and nothing needs to: sharding follows `jax.device_count()`, so
# a demo whose arrays split *is* the SPMD version of itself once the session has
# more than one device. Listed here are the ones that actually split and so are
# worth the second run; they stay in `files` too, and run unsharded there.
#
# A demo belongs here when the *solve* splits, not merely its transforms -- the
# sharded transforms have their own tests under `tests/galerkin`, and a demo
# listed for them alone would suggest a parallelism it does not have.
# `poisson2D_periodic` gives every Fourier wavenumber an independent banded
# system that `TPMatricesWavenumberSolver` factorises per device; `schnakenberg`
# is Fourier in both directions, so its implicit diffusion operator is diagonal
# and the stage solves are elementwise. Neither communicates. `schnakenberg`
# additionally carries a jitted IMEX step, which is where `pin_state` and the
# replicated forcings have to hold up.
#
# Coverage is size-dependent and degrades quietly rather than failing: an extent
# that does not divide by the device count takes the local path and the demo
# still passes, so a green run here is not on its own evidence that anything was
# sharded. Remember to design the demos with appropriate sizes!
#
# To run one by hand rather than through pytest, give the interpreter the devices
# the fixture would have: `JAX_NUM_CPU_DEVICES=2`, or
# `jax.config.update("jax_num_cpu_devices", 2)` *before* the first jaxfun import
# -- `jaxfun.sharding` builds its device mesh at import, and JAX refuses the
# config update once a backend is live.
SPMD_DEMOS = [
    "poisson2D_periodic",
    "schnakenberg",
]

# Demos whose own verification needs float64, so they enable it for themselves at
# import. Precision is a global switch: flipping it once another demo has been
# traced in the same interpreter leaves constants of both kinds in one graph --
# jaxfun's Gauss-Legendre tables are built at whatever precision was active when
# `fastgl` was imported, and a later float64 branch will not match them. So these
# run only when the session is already float64 (`pytest --float64`), where their
# own switch is a no-op. Detected from the source, the way spmd demos are: a
# guarded `if "PYTEST" not in os.environ:` flip is indented and does not count,
# because under pytest it never fires.
float64_files = [
    f.stem
    for f in _all_files
    if '\njax.config.update("jax_enable_x64", True)' in f.read_text()
]

SMOKE_DEMOS = [
    "poisson1D",
]


def _run_demo(demo: str) -> None:
    if demo in float64_files:
        import jax

        if not jax.config.jax_enable_x64:
            pytest.skip(f"{demo} needs float64; run the examples with --float64")
    with contextlib.suppress(SystemExit):
        runpy.run_path(str(demo_paths[demo]), run_name="__main__")


@pytest.mark.smoke
@pytest.mark.parametrize("demo", SMOKE_DEMOS)
def test_demo_smoke(demo: str) -> None:
    _run_demo(demo)


@pytest.mark.slow
@pytest.mark.examples
@pytest.mark.parametrize(
    "demo",
    files,
)
def test_demos(demo: str) -> None:
    _run_demo(demo)


@pytest.mark.slow
@pytest.mark.examples
@pytest.mark.spmd
@pytest.mark.parametrize(
    "demo",
    SPMD_DEMOS,
)
def test_demos_spmd(demo: str) -> None:
    _run_demo(demo)


if __name__ == "__main__":
    test_demos("poisson1D")
