import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.prior import Prior


tfd = tfp.distributions


def test_prior_invalid_distribution_raises():
    with pytest.raises(ValueError):
        Prior("not-a-distribution")


def test_prior_log_prob_inverse_sum_matches_sum():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    state = prior.log_prob_inverse(jp.array(0.0))
    assert jp.isclose(state.log_prob_sum, state.log_prob.sum())


def test_prior_log_prob_matches_distribution():
    distribution = tfd.Normal(0.0, 1.0)
    prior = Prior(distribution, name="x")
    forward_sample = jp.array(0.5)
    state = prior.log_prob(forward_sample)
    expected = distribution.log_prob(forward_sample)
    assert jp.isclose(state.log_prob_sum, expected)


def test_prior_log_prob_inverse_adds_jacobian():
    distribution = tfd.Normal(0.0, 1.0)
    bijector = tfp.bijectors.Exp()
    prior = Prior(distribution, bijector=bijector, name="x")
    inverse_sample = jp.array(0.2)
    forward_sample = bijector(inverse_sample)
    expected = distribution.log_prob(forward_sample)
    expected = expected + bijector.forward_log_det_jacobian(inverse_sample)
    state = prior.log_prob_inverse(inverse_sample)
    assert jp.isclose(state.log_prob_sum, expected)


def test_prior_sample_inverse_single_shape():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    sample = prior.sample_inverse(jax.random.PRNGKey(0), num_samples=1)
    assert sample.shape == ()


def test_prior_sample_batch_shape():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    sample = prior.sample(jax.random.PRNGKey(1), num_samples=3)
    assert sample.shape == (3,)


def test_prior_fit_bijector_returns_prior():
    bijector = tfp.bijectors.Chain(
        [tfp.bijectors.Shift(0.0), tfp.bijectors.Scale(1.0)]
    )
    prior = Prior(tfd.Normal(0.0, 1.0), bijector=bijector, name="x")
    target = tfd.Normal(5.0, 0.2)
    before_value = prior.bijector.inverse(jp.array(1.0))
    new_prior, losses = prior.fit_bijector(
        jax.random.PRNGKey(2),
        target,
        num_samples=256,
        num_steps=50,
        print=False,
        return_losses=True,
    )
    assert len(losses) == 50
    after_value = new_prior.bijector.inverse(jp.array(1.0))
    assert not jp.allclose(before_value, after_value)


def test_prior_fit_bijector_updates_distribution():
    bijector = tfp.bijectors.Chain(
        [tfp.bijectors.Shift(0.0), tfp.bijectors.Scale(1.0)]
    )
    prior = Prior(tfd.Normal(0.0, 1.0), bijector=bijector, name="x")
    target = tfd.Normal(5.0, 0.2)
    fitted_prior = prior.fit_bijector(
        jax.random.PRNGKey(1),
        target,
        num_samples=128,
        num_steps=10,
        print=False,
    )
    assert fitted_prior.distribution is target
