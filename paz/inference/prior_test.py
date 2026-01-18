import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.prior import Prior


tfd = tfp.distributions


def test_prior_invalid_distribution_raises():
    with pytest.raises(ValueError):
        Prior("not-a-distribution")


def test_prior_apply_log_prob_sum_matches_sum():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    state = prior.apply(jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_prior_sample_inverse_single_shape():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    sample = prior.sample_inverse(jax.random.PRNGKey(0), num_samples=1)
    assert sample.shape == ()


def test_prior_sample_batch_shape():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    sample = prior.sample(jax.random.PRNGKey(1), num_samples=3)
    assert sample.shape == (3,)
