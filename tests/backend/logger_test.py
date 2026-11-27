import warnings
import importlib

from paz.backend import logger


def test_logger_import_issues_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(logger)

        assert len(w) >= 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "paz.logger is deprecated" in str(w[-1].message)


def test_logger_re_exports_directory_functions():
    assert hasattr(logger, "make_directory")
    assert hasattr(logger, "make_timestamped_directory")
    assert hasattr(logger, "find_path")


def test_logger_re_exports_file_functions():
    assert hasattr(logger, "write_dictionary")
    assert hasattr(logger, "write_weights")
    assert hasattr(logger, "load_csv")
    assert hasattr(logger, "load_latest")


def test_logger_re_exports_console_functions():
    assert hasattr(logger, "warn")


def test_logger_keeps_setup_function():
    assert hasattr(logger, "setup")
    assert callable(logger.setup)
