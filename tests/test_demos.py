import contextlib
import runpy
from pathlib import Path

import pytest

root = Path(__file__).parent.parent
files = [f.stem for f in root.glob("examples/*.py") if f.is_file()]


@pytest.mark.slow
@pytest.mark.parametrize(
    "demo",
    files,
)
def test_demos(demo: str) -> None:
    with contextlib.suppress(SystemExit):
        runpy.run_path(f"examples/{demo}.py", run_name="__main__")


if __name__ == "__main__":
    test_demos("poisson1D")
