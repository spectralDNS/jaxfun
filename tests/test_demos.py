import contextlib
import runpy

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "demo",
    [
        "poisson1D",
        "poisson1D_curv",
        "poisson1D_spikan",
        "poisson2D",
        "poisson2D_curv",
        "poisson3D",
        "poisson1D_periodic",
        "poisson2D_periodic",
        "poisson2D_parametric",
        "poisson2D_lstsq",
        "biharmonic2D",
        "helmholtz1D",
        "helmholtz2D",
    ],
)
def test_demos(demo: str) -> None:
    with contextlib.suppress(SystemExit):
        runpy.run_path(f"examples/{demo}.py", run_name="__main__")


if __name__ == "__main__":
    test_demos("poisson1D")
