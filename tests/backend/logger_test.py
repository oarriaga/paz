import argparse
import importlib
import json
import os
import warnings

import jax
import jax.numpy as jp

from paz.backend import directory, file, logger, message


def test_logger_import_issues_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(logger)

        assert len(w) >= 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "paz.logger is deprecated" in str(w[-1].message)


def test_logger_re_exports_directory_functions():
    assert logger.make_directory is directory.make
    assert logger.make_timestamped_directory is directory.make_timestamped
    assert logger.find_path is directory.find_latest


def test_logger_re_exports_file_functions():
    assert logger.write_dictionary is file.write_json
    assert logger.write_weights is file.write_weights
    assert logger.load_csv is file.load_csv
    assert logger.load_latest is file.load_latest


def test_logger_re_exports_console_functions():
    assert logger.warn is message.warn


def test_setup_writes_parameters_and_returns_seeded_key(tmp_path):
    args = argparse.Namespace(root=str(tmp_path), seed=3, model="a", label="b")
    experiment_root, key = logger.setup(args)
    with open(os.path.join(experiment_root, "parameters.json")) as filedata:
        parameters = json.load(filedata)
    assert parameters == vars(args)
    assert os.path.dirname(experiment_root) == str(tmp_path)
    assert experiment_root.endswith("_a_b")
    assert jp.array_equal(key, jax.random.PRNGKey(3))
