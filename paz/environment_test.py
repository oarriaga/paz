import os

import jax

import paz


def test_set_memory_fraction_sets_environment():
    previous = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    try:
        result = paz.environment.set_memory_fraction(0.85)
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.85"
        assert result == "0.85"
    finally:
        _restore_environment_variable(
            "XLA_PYTHON_CLIENT_MEM_FRACTION", previous
        )


def test_set_gpu_device_sets_environment():
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        result = paz.environment.set_gpu_device("0")
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        assert result == "0"
    finally:
        _restore_environment_variable("CUDA_VISIBLE_DEVICES", previous)


def test_set_debug_nans_updates_config():
    previous = jax.config.jax_debug_nans
    try:
        result = paz.environment.set_debug_nans(True)
        assert jax.config.jax_debug_nans is True
        assert result is True
    finally:
        paz.environment.set_debug_nans(previous)


def test_set_platform_updates_config():
    previous = jax.config.read("jax_platform_name")
    try:
        result = paz.environment.set_platform("cpu")
        assert jax.config.read("jax_platform_name") == "cpu"
        assert result == "cpu"
    finally:
        paz.environment.set_platform(previous)


def _restore_environment_variable(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
