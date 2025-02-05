import os

import jax


def pytest_addoption(parser):
    parser.addoption("--float64", action="store_true", default=False)


def pytest_configure(config):
    jax.config.update("jax_enable_x64", config.getoption("--float64"))
    os.environ["pytest"] = "True"


def pytest_unconfigure(config):
    del os.environ["pytest"]
