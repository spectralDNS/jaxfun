import contextlib
import runpy
from pathlib import Path

import pytest

root = Path(__file__).parent.parent

_all_files = [f for f in root.glob("examples/*.py") if f.is_file()]
_all_files = [f for f in _all_files if f.stem != "DrivenCavity_lstsq"]

spmd_files = [
    f.stem for f in _all_files if "pytestmark = pytest.mark.spmd" in f.read_text()
]
files = [f.stem for f in _all_files if f.stem not in spmd_files]

SMOKE_DEMOS = [
    "poisson1D",
]


def _run_demo(demo: str) -> None:
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
