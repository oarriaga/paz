import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent import Latent


tfd = tfp.distributions


def test_latent_invalid_callable_raises():
    with pytest.raises(ValueError):
        Latent("not-callable")


def test_latent_log_prob_inverse_sum_matches_sum():
    def distribution_fn(mu):
        return tfd.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    state = latent.log_prob_inverse(jp.array(0.0), jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_latent_log_prob_matches_distribution():
    def distribution_fn(mu):
        return tfd.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    forward_sample = jp.array(0.5)
    mu = jp.array(0.1)
    state = latent.log_prob(forward_sample, mu)
    expected = tfd.Normal(mu, 1.0).log_prob(forward_sample)
    assert jp.isclose(state.log_prob_sum, expected)


def test_latent_log_prob_inverse_adds_jacobian():
    def distribution_fn(mu):
        return tfd.Normal(mu, 1.0)

    bijector = tfp.bijectors.Exp()
    latent = Latent(distribution_fn, bijector=bijector, name="x")()
    inverse_sample = jp.array(0.2)
    mu = jp.array(0.1)
    forward_sample = bijector(inverse_sample)
    expected = tfd.Normal(mu, 1.0).log_prob(forward_sample)
    expected = expected + bijector.forward_log_det_jacobian(inverse_sample)
    state = latent.log_prob_inverse(inverse_sample, mu)
    assert jp.isclose(state.log_prob_sum, expected)


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
