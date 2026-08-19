import contextlib
import runpy
from pathlib import Path

import pytest

root = Path(__file__).parent.parent

# Modules that live in examples/ but are not demos: imported by a demo rather
# than run on their own. `runpy` puts examples/ on sys.path, which is what makes
# those imports resolve. OrrSommerfeld_eigs supplies the eigenmode NavierStokes
# seeds itself with; running it directly only reaches its argparse entry point,
# which rejects pytest's argv and exits before computing anything.
NOT_DEMOS = {"OrrSommerfeld_eigs"}

_all_files = [f for f in root.glob("examples/*.py") if f.is_file()]
_all_files = [
    f for f in _all_files if "DrivenCavity" not in f.stem and f.stem not in NOT_DEMOS
]

spmd_files = [
    f.stem for f in _all_files if "pytestmark = pytest.mark.spmd" in f.read_text()
]
files = [f.stem for f in _all_files if f.stem not in spmd_files]

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
        runpy.run_path(str(root / "examples" / f"{demo}.py"), run_name="__main__")


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
    spmd_files,
)
def test_demos_spmd(demo: str) -> None:
    _run_demo(demo)


if __name__ == "__main__":
    test_demos("poisson1D")
