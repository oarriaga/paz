import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent import Latent


tfd = tfp.distributions


def test_latent_invalid_callable_raises():
    with pytest.raises(ValueError):
        Latent("not-callable")


def test_latent_apply_log_prob_sum_matches_sum():
    def distribution_fn(mu):
        return tfd.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    state = latent.apply(jp.array(0.0), jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_latent_sample_inverse_batch_shape():
    def distribution_fn(mu):
        return tfd.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    sample = latent.sample_inverse(jax.random.PRNGKey(0), 3, jp.array(0.0))
    assert sample.shape == (3,)


def test_latent_sample_single_shape():
    def distribution_fn(mu):
        return tfd.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    sample = latent.sample(jax.random.PRNGKey(1), 1, jp.array(0.0))
    assert sample.shape == ()
