import pytest

from paz.inference import naming


def test_to_snake_case_basic():
    assert naming._to_snake_case("MyName") == "my_name"


def test_to_snake_case_none():
    assert naming._to_snake_case(None) == ""


def test_callable_name_lambda():
    assert naming._callable_name(lambda x: x) == "lambda"


def test_callable_name_function():
    def custom_name():
        return None

    assert naming._callable_name(custom_name) == "custom_name"


def test_build_auto_name_base_first():
    naming._NAME_COUNTERS.clear()
    assert naming.build_auto_name("prior", "Normal") == "prior_normal_0"


def test_build_auto_name_base_second():
    naming._NAME_COUNTERS.clear()
    naming.build_auto_name("prior", "Normal")
    assert naming.build_auto_name("prior", "Normal") == "prior_normal_1"


def test_build_auto_name_role_only():
    naming._NAME_COUNTERS.clear()
    assert naming.build_auto_name("latent") == "latent_0"


def test_build_prior_name_uses_distribution():
    naming._NAME_COUNTERS.clear()

    class DummyNormal:
        pass

    distribution = DummyNormal()
    assert naming.build_prior_name(distribution) == "prior_dummy_normal_0"


def test_build_latent_name_uses_callable():
    naming._NAME_COUNTERS.clear()

    def latent_fn():
        return None

    assert naming.build_latent_name(latent_fn) == "latent_latent_fn_0"


def test_build_observation_name_uses_callable():
    naming._NAME_COUNTERS.clear()

    def obs_fn():
        return None

    assert naming.build_observation_name(obs_fn) == "observation_obs_fn_0"


def test_build_auto_name_role_none_base():
    naming._NAME_COUNTERS.clear()
    assert naming.build_auto_name(None, None) == "_0"
