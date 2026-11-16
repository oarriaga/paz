import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.observable import Observable


def test_observable_invalid_callable_raises():
    with pytest.raises(ValueError):
        Observable("not-callable")


def test_observable_log_prob_sum_matches_sum():
    def likelihood(mu):
        return paz.distributions.Normal(mu, 1.0)

    obs = Observable(likelihood, name="y")(jp.array(0.0))
    state = obs.log_prob(jp.array(0.5), jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_observable_log_prob_matches_distribution():
    def likelihood(mu):
        return paz.distributions.Normal(mu, 1.0)

    obs = Observable(likelihood, name="y")(jp.array(0.0))
    observation = jp.array(0.5)
    mu = jp.array(0.0)
    state = obs.log_prob(observation, mu)
    expected = paz.distributions.Normal(mu, 1.0).log_prob(observation)
    assert jp.isclose(state.log_prob_sum, expected)


def test_observable_log_prob_inverse_is_none():
    def likelihood(mu):
        return paz.distributions.Normal(mu, 1.0)

    obs = Observable(likelihood, name="y")(jp.array(0.0))
    assert obs.log_prob_inverse is None


def test_observable_sample_shape_batch():
    def likelihood(mu):
        return paz.distributions.Normal(mu, 1.0)

    obs = Observable(likelihood, name="y")(jp.array(0.0))
    sample = obs.sample(jax.random.PRNGKey(0), 4, jp.array(0.0))
    assert sample.shape == (4,)
