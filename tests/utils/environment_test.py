import jax
import paz.utils.environment as utils_environment

def test_set_debug_nans_updates_config():
    previous = jax.config.jax_debug_nans
    try:
        result = utils_environment.set_debug_nans(True)
        assert jax.config.jax_debug_nans is True
        assert result is True
    finally:
        utils_environment.set_debug_nans(previous)


def test_set_platform_updates_config():
    previous = jax.config.read("jax_platform_name")
    try:
        result = utils_environment.set_platform("cpu")
        assert jax.config.read("jax_platform_name") == "cpu"
        assert result == "cpu"
    finally:
        utils_environment.set_platform(previous)
