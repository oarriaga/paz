from functools import partial

import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.latent import Latent
from paz.inference.pgm import marginalize, recover_discrete_posterior
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior
from paz.inference.types import SampleType


def build_scalar_mixture_model(
    mu0_value, mu1_value, sigma_value, p_value, y_value
):
    mu0 = Prior(paz.distributions.Deterministic(mu0_value), name="mu0")
    mu1 = Prior(paz.distributions.Deterministic(mu1_value), name="mu1")
    sigma = Prior(paz.distributions.Deterministic(sigma_value), name="sigma")
    p = Prior(paz.distributions.Deterministic(p_value), name="p")

    def z_distribution(p):
        return paz.distributions.Bernoulli(probs=p)

    z = Latent(z_distribution, name="z")(p)

    def y_distribution(z, mu0, mu1, sigma):
        mean = jp.where(z == 1, mu1, mu0)
        return paz.distributions.Normal(mean, sigma)

    y_obs = Observable(y_distribution, name="y_obs")(
        z, mu0, mu1, sigma
    )
    data = {"y_obs": jp.array(y_value)}
    return PGM([mu0, mu1, sigma, p], [y_obs], "scalar_mixture"), data


def make_theta_inverse_samples(mu0_value, mu1_value, sigma_value, p_value):
    Sample = SampleType(["mu0", "mu1", "sigma", "p"])
    return Sample(mu0_value, mu1_value, sigma_value, p_value)


def compute_normal_log_pdf(mean, stdv, value):
    z = (value - mean) / stdv
    return -0.5 * z**2 - 0.5 * jp.log(2.0 * jp.pi) - jp.log(stdv)


def compute_analytic_log_marginal(*values):
    log_joint0, log_joint1 = compute_analytic_log_joints(*values)
    return jp.logaddexp(log_joint0, log_joint1)


def compute_analytic_log_joints(mu0_value, mu1_value, sigma_value, p_value, y):
    log_prob0 = compute_normal_log_pdf(mu0_value, sigma_value, y)
    log_prob1 = compute_normal_log_pdf(mu1_value, sigma_value, y)
    return jp.log1p(-p_value) + log_prob0, jp.log(p_value) + log_prob1


def compute_analytic_posterior(*values):
    logits = jp.stack(compute_analytic_log_joints(*values))
    return jp.exp(logits - jp.logaddexp(logits[0], logits[1]))


def build_bijected_model(beta):
    p = Prior(beta, paz.bijectors.Sigmoid(), "p")

    def z_distribution(p):
        return paz.distributions.Bernoulli(probs=p)

    z = Latent(z_distribution, name="z")(p)

    def y_distribution(z):
        return paz.distributions.Normal(z, 1.0)

    y_obs = Observable(y_distribution, name="y_obs")(z)
    return PGM([p], [y_obs], "bijected")


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
    prior_state = pgm_marg.prior.log_prob_inverse(theta)
    likelihood_state = pgm_marg.likelihood.log_prob_inverse(theta, data)
    log_prob_sum = prior_state.log_prob_sum + likelihood_state.log_prob_sum
    expected = compute_analytic_log_marginal(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    assert jp.allclose(log_prob_sum, expected)


def test_recover_discrete_posterior_matches_analytic():
    mu0_value, mu1_value = 0.0, 2.0
    sigma_value, p_value, y_value = 1.0, 0.3, 1.5
    pgm, data = build_scalar_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    theta = make_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    posterior = recover_discrete_posterior(pgm, "z", theta, data).posterior
    expected = compute_analytic_posterior(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    assert jp.allclose(posterior, expected)


def build_batched_theta(num_samples):
    values = [jp.full((num_samples,), value) for value in (0.0, 2.0, 1.0, 0.3)]
    return make_theta_inverse_samples(*values)


def test_marginalize_log_prob_vmaps_over_samples():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    pgm_marg = marginalize(pgm, ["z"])
    log_prob_inverse = partial(pgm_marg.log_prob_inverse, data=data)
    log_probs = jax.vmap(log_prob_inverse)(build_batched_theta(4))
    expected = compute_analytic_log_marginal(0.0, 2.0, 1.0, 0.3, 1.5)
    assert jp.allclose(log_probs, jp.full((4,), expected))


def test_recover_discrete_posterior_vmaps_over_samples():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    recover = partial(recover_discrete_posterior, pgm, "z", data=data)
    posterior = jax.vmap(recover)(build_batched_theta(4)).posterior
    expected = compute_analytic_posterior(0.0, 2.0, 1.0, 0.3, 1.5)
    assert jp.allclose(posterior, jp.tile(expected, (4, 1)))


def test_marginalize_multiple_names_raises():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z", "x"])


def test_marginalize_nonfinite_discrete_raises():
    z = Prior(paz.distributions.Poisson(1.0), name="z")
    pgm = PGM([z], [z], "poisson")
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z"])


def test_marginalize_vector_discrete_raises():
    z = Prior(paz.distributions.Bernoulli(logits=jp.zeros((2,))), name="z")
    pgm = PGM([z], [z], "vector_bernoulli")
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z"])


def test_marginalize_discrete_non_identity_bijector_raises():
    bernoulli = paz.distributions.Bernoulli(0.5)
    z = Prior(bernoulli, name="z", bijector=paz.bijectors.Sigmoid())
    pgm = PGM([z], [z], "bijected_bernoulli")
    with pytest.raises(NotImplementedError):
        marginalize(pgm, ["z"])


def test_marginalize_joint_log_prob_matches_analytic():
    mu0_value, mu1_value = 0.0, 2.0
    sigma_value, p_value, y_value = 1.0, 0.3, 1.5
    pgm, data = build_scalar_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    pgm_marg = marginalize(pgm, ["z"])
    theta = make_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    log_density = pgm_marg.log_prob_inverse(theta, data)
    expected = compute_analytic_log_marginal(
        mu0_value, mu1_value, sigma_value, p_value, y_value
    )
    assert jp.allclose(log_density, expected)


def test_marginalize_forward_prior_log_prob_excludes_jacobian():
    beta = paz.distributions.Beta(2.0, 2.0)
    prior = marginalize(build_bijected_model(beta), ["z"]).prior
    log_prob_sum = prior.log_prob({"p": jp.array(0.3)}).log_prob_sum
    assert jp.allclose(log_prob_sum, beta.log_prob(0.3))


def test_marginalize_without_prior_mass_is_not_nan():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    pgm_marg = marginalize(pgm, ["z"])
    theta = make_theta_inverse_samples(0.0, 2.0, 1.0, 0.5)
    likelihood_state = pgm_marg.likelihood.log_prob_inverse(theta, data)
    assert jp.isfinite(likelihood_state.log_prob_sum)
    assert jp.isneginf(pgm_marg.log_prob_inverse(theta, data))


def test_marginalize_keeps_prior_of_remaining_latents():
    beta = paz.distributions.Beta(2.0, 2.0)
    pgm = build_bijected_model(beta)
    samples = {"p": jp.array(0.4)}
    state = marginalize(pgm, ["z"]).prior.log_prob_inverse(samples)
    expected = pgm.inputs[0].log_prob_inverse(samples["p"]).log_prob_sum
    assert jp.allclose(state.log_prob_sum, expected)


def test_marginalize_samples_without_discrete_node():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    samples = marginalize(pgm, ["z"]).sample(jax.random.PRNGKey(0), 2000)
    assert set(samples._fields) == {"mu0", "mu1", "sigma", "p", "y_obs"}
    assert jp.isclose(samples.y_obs.mean(), 0.3 * 2.0, atol=0.1)


def test_marginalize_latent_child_raises():
    z = Prior(paz.distributions.Bernoulli(probs=0.5), name="z")
    mean = Latent(lambda z: paz.distributions.Normal(z, 1.0), name="mean")(z)
    y = Observable(lambda m: paz.distributions.Normal(m, 1.0), name="y")(mean)
    with pytest.raises(NotImplementedError):
        marginalize(PGM([z], [y], "latent_child"), ["z"])


def test_marginalize_two_children_raises():
    z = Prior(paz.distributions.Bernoulli(probs=0.5), name="z")
    y0 = Observable(lambda z: paz.distributions.Normal(z, 1.0), name="y0")(z)
    y1 = Observable(lambda z: paz.distributions.Normal(z, 2.0), name="y1")(z)
    with pytest.raises(NotImplementedError):
        marginalize(PGM([z], [y0, y1], "two_children"), ["z"])


def test_marginalized_model_infers():
    data = {"y_obs": jp.array([1.5, -0.5, 0.2])}
    beta = paz.distributions.Beta(2.0, 2.0)
    pgm_marg = marginalize(build_bijected_model(beta), ["z"])
    kwargs = {"num_samples": 20, "num_chains": 2, "sigma": 0.2}
    posterior = pgm_marg.infer(jax.random.PRNGKey(0), data, **kwargs)
    position = jax.tree.map(lambda leaf: leaf[-1, 0], posterior.inverse_samples)
    expected = pgm_marg.log_prob_inverse(position, data)
    assert posterior.samples.p.shape == (20, 2)
    assert jp.allclose(posterior.inverse_log_probs[-1, 0], expected)


def test_marginalize_samples_child_of_parentless_discrete_prior():
    def y_distribution(z):
        return paz.distributions.Normal(2.0 * z, 1.0)

    z = Prior(paz.distributions.Bernoulli(probs=0.3), name="z")
    y = Observable(y_distribution, name="y")(z)
    pgm_marg = marginalize(PGM([z], [y], "only_discrete"), ["z"])
    samples = pgm_marg.sample(jax.random.PRNGKey(0), 4000)
    assert samples._fields == ("y",)
    assert jp.isclose(samples.y.mean(), 0.6, atol=0.1)


def test_recover_discrete_posterior_missing_data_raises():
    pgm, data = build_scalar_mixture_model(0.0, 2.0, 1.0, 0.3, 1.5)
    theta = make_theta_inverse_samples(0.0, 2.0, 1.0, 0.3)
    with pytest.raises(ValueError):
        recover_discrete_posterior(pgm, "z", theta, {})


def test_marginalize_child_sees_float_values_when_sampling():
    def y_distribution(z):
        assert jp.issubdtype(z.dtype, jp.floating)
        return paz.distributions.Normal(z, 1.0)

    z = Prior(paz.distributions.Bernoulli(probs=0.3), name="z")
    y = Observable(y_distribution, name="y")(z)
    pgm_marg = marginalize(PGM([z], [y], "integer_discrete"), ["z"])
    pgm_marg.sample(jax.random.PRNGKey(0), 3)
    pgm_marg.likelihood.log_prob({}, {"y": 0.5})
