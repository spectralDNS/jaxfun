import os

import jax


def pytest_addoption(parser) -> None:
    parser.addoption("--float64", action="store_true", default=False)


def pytest_configure(config) -> None:
    jax.config.update("jax_enable_x64", config.getoption("--float64"))
    os.environ["PYTEST"] = "True"


def pytest_unconfigure(config) -> None:
    del os.environ["PYTEST"]
