import contextlib
import runpy

demos = [
    "poisson1D",
    "poisson1D_curv",
    "poisson2D",
    "poisson2D_curv",
    "poisson3D",
    "poisson1D_periodic",
    "poisson2D_periodic",
    "poisson2D_parametric",
    "biharmonic2D",
    "helmholtz2D",
]


def test_demos():
    for demo in demos:
        with contextlib.suppress(SystemExit):
            runpy.run_path(f"examples/{demo}.py", run_name="__main__")


if __name__ == "__main__":
    test_demos()
