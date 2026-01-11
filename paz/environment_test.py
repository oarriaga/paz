import jax
import paz.environment

import paz


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
