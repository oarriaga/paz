import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="run heavy torch/compute tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "heavy: torch/compute-heavy tests; run with --run-heavy"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-heavy"):
        return
    skip_heavy = pytest.mark.skip(reason="need --run-heavy to run")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
