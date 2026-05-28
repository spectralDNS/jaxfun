import os

import pytest


@pytest.fixture(autouse=True, scope="module")
def clear_jax_caches_after_module():
    yield

    import jax

    jax.clear_caches()


def pytest_addoption(parser) -> None:
    parser.addoption("--float64", action="store_true", default=False)
    parser.addoption(
        "--num-devices",
        type=int,
        default=1,
        help="Number of JAX CPU devices to expose (must be >1 to run spmd-marked tests).",  # noqa: E501
    )


def pytest_configure(config) -> None:
    os.environ["JAX_ENABLE_X64"] = "1" if config.getoption("--float64") else "0"
    import jax

    jax.config.update("jax_enable_x64", config.getoption("--float64"))
    n = config.getoption("--num-devices")
    if n > 1:
        jax.config.update("jax_num_cpu_devices", n)
    os.environ["PYTEST"] = "True"


def pytest_collection_modifyitems(config, items) -> None:
    n = config.getoption("--num-devices")
    if n > 1:
        selected = [item for item in items if "spmd" in item.keywords]
        deselected = [item for item in items if "spmd" not in item.keywords]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_runtest_setup(item) -> None:
    if "spmd" in item.keywords:
        import jax

        if jax.device_count() < 2:
            pytest.skip("spmd tests require --num-devices=2 (or more)")


def pytest_unconfigure(config) -> None:
    del os.environ["PYTEST"]
