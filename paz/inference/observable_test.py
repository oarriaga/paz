import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.observable import Observable


tfd = tfp.distributions


def test_observable_invalid_callable_raises():
    with pytest.raises(ValueError):
        Observable("not-callable")


def test_observable_apply_log_prob_sum_matches_sum():
    def likelihood(mu):
        return tfd.Normal(mu, 1.0)

    obs = Observable(likelihood, name="y")(jp.array(0.0))
    state = obs.apply(jp.array(0.5), jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_observable_sample_shape_batch():
    def likelihood(mu):
        return tfd.Normal(mu, 1.0)

    obs = Observable(likelihood, name="y")(jp.array(0.0))
    sample = obs.sample(jax.random.PRNGKey(0), 4, jp.array(0.0))
    assert sample.shape == (4,)
