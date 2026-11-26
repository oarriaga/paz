import os

import pytest


def pytest_configure(config):
    line = "weights: test downloads pretrained weights or assets"
    config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PAZ_WEIGHTS_TEST") != "1":
        reason = "set PAZ_WEIGHTS_TEST=1 to run tests that download weights"
        for item in items:
            if "weights" in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
