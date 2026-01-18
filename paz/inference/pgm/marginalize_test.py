import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent import Latent
from paz.inference.pgm import marginalize, recover_discrete_posterior
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior
from paz.inference.types import SampleType

tfd = tfp.distributions
tfb = tfp.bijectors


def build_scalar_mixture_model(
    mu0_value, mu1_value, sigma_value, p_value, y_value
):
    mu0 = Prior(tfd.Deterministic(mu0_value), name="mu0")
    mu1 = Prior(tfd.Deterministic(mu1_value), name="mu1")
    sigma = Prior(tfd.Deterministic(sigma_value), name="sigma")
    p = Prior(tfd.Deterministic(p_value), name="p")

    def z_distribution(p):
        return tfd.Bernoulli(probs=p)

    z = Latent(z_distribution, name="z")(p)

    def y_distribution(z, mu0, mu1, sigma):
        mean = jp.where(z == 1, mu1, mu0)
        return tfd.Normal(mean, sigma)

    y_obs = Observable(y_distribution, name="y_obs")(
        z, mu0, mu1, sigma
    )
    data = {"y_obs": jp.array(y_value)}
    return PGM([mu0, mu1, sigma, p], [y_obs], "scalar_mixture"), data


def make_theta_inverse_samples(mu0_value, mu1_value, sigma_value, p_value):
    Sample = SampleType(["mu0", "mu1", "sigma", "p"])
    return Sample(mu0_value, mu1_value, sigma_value, p_value)


def compute_analytic_log_marginal(mu0_value, mu1_value, sigma_value, p_value, y_value):
    log_prob0 = tfd.Normal(mu0_value, sigma_value).log_prob(y_value)
    log_prob1 = tfd.Normal(mu1_value, sigma_value).log_prob(y_value)
    log_joint0 = jp.log1p(-p_value) + log_prob0
    log_joint1 = jp.log(p_value) + log_prob1
    return jp.logaddexp(log_joint0, log_joint1)


def compute_analytic_posterior(mu0_value, mu1_value, sigma_value, p_value, y_value):
    log_prob0 = tfd.Normal(mu0_value, sigma_value).log_prob(y_value)
    log_prob1 = tfd.Normal(mu1_value, sigma_value).log_prob(y_value)
    logits = jp.stack([jp.log1p(-p_value) + log_prob0, jp.log(p_value) + log_prob1])
    return jp.exp(logits - jp.logaddexp(logits[0], logits[1]))


def test_marginalize_log_prob_matches_analytic():
    mu0_value, mu1_value = 0.0, 2.0
    sigma_value, p_value, y_value = 1.0, 0.3, 1.5
    pgm, data = build_scalar_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    pgm_marg = marginalize(pgm, ["z"])
    theta = make_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    state = pgm_marg.apply(theta, data)
    expected = compute_analytic_log_marginal(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    assert jp.allclose(state.log_prob_sum, expected)


def test_recover_discrete_posterior_matches_analytic():
    mu0_value, mu1_value = 0.0, 2.0
    sigma_value, p_value, y_value = 1.0, 0.3, 1.5
    pgm, data = build_scalar_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    pgm_marg = marginalize(pgm, ["z"])
    theta = make_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    posterior = recover_discrete_posterior(
        pgm_marg, "z", theta, data
    ).posterior
    expected = compute_analytic_posterior(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    assert jp.allclose(posterior, expected)


def test_marginalize_batched_log_prob_shape():
    num_samples = 4
    mu0_value, mu1_value = 0.0, 2.0
    sigma_value, p_value, y_value = 1.0, 0.3, 1.5
    pgm, data = build_scalar_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    pgm_marg = marginalize(pgm, ["z"])
    theta = make_theta_inverse_samples(
        jp.full((num_samples,), mu0_value),
        jp.full((num_samples,), mu1_value),
        jp.full((num_samples,), sigma_value),
        jp.full((num_samples,), p_value),
    )
    state = pgm_marg.apply(theta, data)
    assert state.log_prob_sum.shape == (num_samples,)


def test_recover_discrete_posterior_batched_shape():
    num_samples = 4
    mu0_value, mu1_value = 0.0, 2.0
    sigma_value, p_value, y_value = 1.0, 0.3, 1.5
    pgm, data = build_scalar_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    pgm_marg = marginalize(pgm, ["z"])
    theta = make_theta_inverse_samples(
        jp.full((num_samples,), mu0_value),
        jp.full((num_samples,), mu1_value),
        jp.full((num_samples,), sigma_value),
        jp.full((num_samples,), p_value),
    )
    posterior = recover_discrete_posterior(
        pgm_marg, "z", theta, data
    ).posterior
    assert posterior.shape == (num_samples, 2)


def test_marginalize_multiple_names_raises():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z", "x"])


def test_marginalize_nonfinite_discrete_raises():
    z = Prior(tfd.Poisson(1.0), name="z")
    pgm = PGM([z], [z], "poisson")
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z"])


def test_marginalize_vector_discrete_raises():
    z = Prior(tfd.Bernoulli(logits=jp.zeros((2,))), name="z")
    pgm = PGM([z], [z], "vector_bernoulli")
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z"])


def test_marginalize_discrete_non_identity_bijector_raises():
    z = Prior(tfd.Bernoulli(0.5), name="z", bijector=tfb.Sigmoid())
    pgm = PGM([z], [z], "bijected_bernoulli")
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z"])
