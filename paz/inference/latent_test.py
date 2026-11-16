import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.latent import Latent


def test_latent_invalid_callable_raises():
    with pytest.raises(ValueError):
        Latent("not-callable")


def test_latent_log_prob_inverse_sum_matches_sum():
    def distribution_fn(mu):
        return paz.distributions.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    state = latent.log_prob_inverse(jp.array(0.0), jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_latent_log_prob_matches_distribution():
    def distribution_fn(mu):
        return paz.distributions.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    forward_sample = jp.array(0.5)
    mu = jp.array(0.1)
    state = latent.log_prob(forward_sample, mu)
    expected = paz.distributions.Normal(mu, 1.0).log_prob(forward_sample)
    assert jp.isclose(state.log_prob_sum, expected)


def test_latent_log_prob_inverse_adds_jacobian():
    def distribution_fn(mu):
        return paz.distributions.Normal(mu, 1.0)

    bijector = paz.bijectors.Exp()
    latent = Latent(distribution_fn, bijector=bijector, name="x")()
    inverse_sample = jp.array(0.2)
    mu = jp.array(0.1)
    forward_sample = bijector(inverse_sample)
    expected = paz.distributions.Normal(mu, 1.0).log_prob(forward_sample)
    expected = expected + bijector.forward_log_det_jacobian(inverse_sample)
    state = latent.log_prob_inverse(inverse_sample, mu)
    assert jp.isclose(state.log_prob_sum, expected)


def test_latent_log_det_covers_vector_event():
    def distribution_fn(mu):
        normal = paz.distributions.Normal(mu * jp.ones(5), 1.0)
        return paz.distributions.Independent(normal, 1)

    latent = Latent(distribution_fn, bijector=paz.bijectors.Exp(), name="x")()
    inverse_sample, mu = jp.linspace(-1.0, 1.0, 5), jp.array(0.1)
    forward_sample = jp.exp(inverse_sample)
    expected = distribution_fn(mu).log_prob(forward_sample)
    expected = expected + inverse_sample.sum()
    state = latent.log_prob_inverse(inverse_sample, mu)
    assert state.log_prob.shape == ()
    assert jp.isclose(state.log_prob_sum, expected)


def test_latent_sample_inverse_batch_shape():
    def distribution_fn(mu):
        return paz.distributions.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    sample = latent.sample_inverse(jax.random.PRNGKey(0), 3, jp.array(0.0))
    assert sample.shape == (3,)


def test_latent_sample_single_shape():
    def distribution_fn(mu):
        return paz.distributions.Normal(mu, 1.0)

    latent = Latent(distribution_fn, name="x")()
    sample = latent.sample(jax.random.PRNGKey(1), 1, jp.array(0.0))
    assert sample.shape == ()
