import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.latent import Latent
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


def test_observable_builder_called_twice_keeps_separate_edges():
    def likelihood(mu):
        return paz.distributions.Normal(mu, 1.0)

    parent_a = Latent(lambda: paz.distributions.Normal(0.0, 1.0), name="a")()
    parent_b = Latent(lambda: paz.distributions.Normal(0.0, 1.0), name="b")()
    build_observable = Observable(likelihood, name="y")
    observable_a = build_observable(parent_a)
    observable_b = build_observable(parent_b)
    assert [edge.name for edge in observable_a.edges] == ["a"]
    assert [edge.name for edge in observable_b.edges] == ["b"]
