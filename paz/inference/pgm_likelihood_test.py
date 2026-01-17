import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior
from paz.inference.latent_space import to_forward_samples

tfd = tfp.distributions
tfb = tfp.bijectors


def build_bijected_prior_model():
    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    x = Prior(tfd.Uniform(low, high), bijector=bijector, name="x")
    return PGM([x], [x], "single_prior"), bijector, (low, high)


def build_two_observables_model():
    obs1 = jp.array([0.5, 0.6, 0.7])
    obs2 = jp.array([1.5, 1.6, 1.7])

    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])

    mean = Prior(tfd.Normal(0.0, 1.0), name="mean")
    stdv = Prior(tfd.Uniform(low, high), bijector=bijector, name="stdv")

    def likelihood1(mean, stdv):
        return tfd.Normal(mean, stdv)

    def likelihood2(mean, stdv):
        return tfd.Normal(mean + 1, stdv)

    y1 = Observable(likelihood1, name="y1")(mean, stdv)
    y2 = Observable(likelihood2, name="y2")(mean, stdv)
    data = {"y1": obs1, "y2": obs2}
    return PGM([mean, stdv], [y1, y2], "two_observables"), data


def build_single_observable_model():
    observations = jp.array([0.5, 0.6, 0.7])
    mean = Prior(tfd.Normal(0.0, 1.0), name="mean")

    def likelihood(mean):
        return tfd.Normal(mean, 1.0)

    y = Observable(likelihood, name="y")(mean)
    return PGM([mean], [y], "single_observable"), observations


def test_prior_log_prob_inv_matches_apply():
    model, _, _ = build_bijected_prior_model()
    key = jax.random.PRNGKey(0)
    theta_inv = model.sample_inverse(key)
    state = model.apply(theta_inv)
    log_prob = model.prior.log_prob(theta_inv, space="inv")
    assert jp.isclose(log_prob, state.log_prob_sum)


def test_prior_log_prob_fwd_matches_distribution():
    model, bijector, (low, high) = build_bijected_prior_model()
    key = jax.random.PRNGKey(1)
    theta_inv = model.prior.sample(key, space="inv")
    theta_fwd = to_forward_samples(model.prior.latent_space, theta_inv)
    expected_fwd = tfd.Uniform(low, high).log_prob(theta_fwd.x)
    actual_fwd = model.prior.log_prob(theta_fwd, space="fwd")
    actual_inv = model.prior.log_prob(theta_inv, space="inv")
    expected_inv = expected_fwd + bijector.forward_log_det_jacobian(theta_inv.x)
    assert jp.isclose(actual_fwd, expected_fwd)
    assert jp.isclose(actual_inv, expected_inv)


def test_prior_sample_forward_respects_bounds():
    model, _, (low, high) = build_bijected_prior_model()
    key = jax.random.PRNGKey(2)
    samples = model.prior.sample(key, num_samples=1000, space="fwd")
    assert samples.x.min() >= low
    assert samples.x.max() <= high


def test_likelihood_log_prob_matches_apply_observables():
    model, data = build_two_observables_model()
    key = jax.random.PRNGKey(3)
    theta_inv = model.sample_inverse(key)
    state = model.apply(theta_inv, data)
    expected = state.log_prob["y1"] + state.log_prob["y2"]
    actual = model.likelihood.log_prob(theta_inv, data, space="inv")
    assert jp.isclose(actual, expected)


def test_likelihood_accepts_forward_samples_and_list_data():
    model, data = build_two_observables_model()
    key = jax.random.PRNGKey(4)
    theta_inv = model.prior.sample(key, space="inv")
    theta_fwd = to_forward_samples(model.prior.latent_space, theta_inv)
    data_list = [data["y1"], data["y2"]]
    inv_log_prob = model.likelihood.log_prob(theta_inv, data, space="inv")
    fwd_log_prob = model.likelihood.log_prob(theta_fwd, data_list, space="fwd")
    assert jp.isclose(inv_log_prob, fwd_log_prob)


def test_likelihood_accepts_single_output_value():
    model, observations = build_single_observable_model()
    key = jax.random.PRNGKey(5)
    theta_inv = model.sample_inverse(key)
    expected = model.likelihood.log_prob(
        theta_inv, {"y": observations}, space="inv"
    )
    actual = model.likelihood.log_prob(theta_inv, observations, space="inv")
    assert jp.isclose(actual, expected)


def test_likelihood_rejects_single_value_for_multi_output():
    model, data = build_two_observables_model()
    key = jax.random.PRNGKey(6)
    theta_inv = model.sample_inverse(key)
    with pytest.raises(TypeError):
        model.likelihood.log_prob(theta_inv, data["y1"], space="inv")
