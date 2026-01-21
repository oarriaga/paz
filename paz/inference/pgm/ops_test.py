import jax
import jax.numpy as jp
import paz
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent_space import to_forward_samples
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior


tfd = tfp.distributions
tfb = tfp.bijectors


def test_sample_inverse_no_double_sampling():
    num_groups = 3

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), jp.abs(sigma)),
            reinterpreted_batch_ndims=1,
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "test")

    key = jax.random.PRNGKey(0)
    inverse_sample = model.sample_inverse(key, num_samples=1)

    assert hasattr(inverse_sample, "mu")
    assert hasattr(inverse_sample, "sigma")
    assert hasattr(inverse_sample, "groups")
    assert not hasattr(inverse_sample, "y_obs")

    assert inverse_sample.mu.shape == ()
    assert inverse_sample.sigma.shape == ()
    assert inverse_sample.groups.shape == (num_groups,)


def test_sample_forward_all_nodes():
    num_groups = 3

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), jp.abs(sigma)),
            reinterpreted_batch_ndims=1,
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "test")

    key = jax.random.PRNGKey(0)
    forward_sample = model.sample(key, num_samples=1)

    assert hasattr(forward_sample, "mu")
    assert hasattr(forward_sample, "sigma")
    assert hasattr(forward_sample, "groups")
    assert hasattr(forward_sample, "y_obs")

    assert forward_sample.mu.shape == ()
    assert forward_sample.sigma.shape == ()
    assert forward_sample.groups.shape == (num_groups,)
    assert forward_sample.y_obs.shape == (num_groups,)


def test_filtering_consistency_batched():
    num_groups = 3
    num_samples = 10

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), jp.abs(sigma)),
            reinterpreted_batch_ndims=1,
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "test")

    key = jax.random.PRNGKey(0)

    inverse_samples = model.sample_inverse(key, num_samples=num_samples)
    assert inverse_samples.mu.shape == (num_samples,)
    assert inverse_samples.sigma.shape == (num_samples,)
    assert inverse_samples.groups.shape == (num_samples, num_groups)

    key, subkey = jax.random.split(key)
    forward_samples = model.sample(subkey, num_samples=num_samples)
    assert forward_samples.mu.shape == (num_samples,)
    assert forward_samples.sigma.shape == (num_samples,)
    assert forward_samples.groups.shape == (num_samples, num_groups)
    assert forward_samples.y_obs.shape == (num_samples, num_groups)


def test_hierarchical_regression_filtering():
    num_groups = 5

    def slope_prior(mu_slope, sigma_slope):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_slope), jp.abs(sigma_slope)),
            reinterpreted_batch_ndims=1,
        )

    def intercept_prior(mu_intercept, sigma_intercept):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_intercept), jp.abs(sigma_intercept)),
            reinterpreted_batch_ndims=1,
        )

    mu_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_slope")
    mu_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_intercept")
    sigma_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma_slope")
    sigma_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma_intercept")

    slopes = paz.Latent(slope_prior, name="slopes")(mu_slope, sigma_slope)
    intercepts = paz.Latent(intercept_prior, name="intercepts")(
        mu_intercept, sigma_intercept
    )

    X = jp.linspace(0, 1, 50)
    group_idx = jp.repeat(jp.arange(num_groups), 10)

    def likelihood(slopes_val, intercepts_val):
        means = slopes_val[group_idx] * X + intercepts_val[group_idx]
        return tfd.Normal(means, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(slopes, intercepts)
    priors = [mu_slope, mu_intercept, sigma_slope, sigma_intercept]
    model = paz.PGM(priors, [y_obs], "hierarchical_regression")

    key = jax.random.PRNGKey(0)
    inverse_sample = model.sample_inverse(key, num_samples=1)
    assert hasattr(inverse_sample, "slopes")
    assert hasattr(inverse_sample, "intercepts")

    key, subkey = jax.random.split(key)
    forward_sample = model.sample(subkey, num_samples=1)
    assert hasattr(forward_sample, "y_obs")


def test_sample_forward_simple_model():
    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    stdv = paz.Prior(tfd.Normal(0.0, 1.0), name="stdv")

    def likelihood(mu, sigma):
        return tfd.Normal(mu, jp.abs(sigma))

    y_obs = paz.Observable(likelihood, name="y_obs")(mean, stdv)
    model = paz.PGM([mean, stdv], [y_obs], "simple")

    key = jax.random.PRNGKey(0)
    sample = model.sample(key, num_samples=1)

    assert hasattr(sample, "mean")
    assert hasattr(sample, "stdv")
    assert hasattr(sample, "y_obs")
    assert sample.mean.shape == ()
    assert sample.stdv.shape == ()
    assert sample.y_obs.shape == ()


def test_sample_forward_simple_model_batched():
    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    stdv = paz.Prior(tfd.Normal(0.0, 1.0), name="stdv")

    def likelihood(mu, sigma):
        return tfd.Normal(mu, jp.abs(sigma))

    y_obs = paz.Observable(likelihood, name="y_obs")(mean, stdv)
    model = paz.PGM([mean, stdv], [y_obs], "simple")

    key = jax.random.PRNGKey(0)
    num_samples = 10
    sample = model.sample(key, num_samples=num_samples)

    assert sample.mean.shape == (num_samples,)
    assert sample.stdv.shape == (num_samples,)
    assert sample.y_obs.shape == (num_samples,)


def test_sample_forward_hierarchical_single_sample():
    num_groups = 3

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), sigma),
            reinterpreted_batch_ndims=1,
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "hierarchical")

    key = jax.random.PRNGKey(0)
    sample = model.sample(key, num_samples=1)

    assert sample.mu.shape == ()
    assert sample.sigma.shape == ()
    assert sample.groups.shape == (num_groups,)
    assert sample.y_obs.shape == (num_groups,)


def test_sample_forward_hierarchical_batched():
    num_groups = 3
    num_samples = 10

    def group_prior(mu, sigma):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu), sigma),
            reinterpreted_batch_ndims=1,
        )

    mu = paz.Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma")
    groups = paz.Latent(group_prior, name="groups")(mu, sigma)

    def likelihood(group_params):
        return tfd.Normal(group_params, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(groups)
    model = paz.PGM([mu, sigma], [y_obs], "hierarchical")

    key = jax.random.PRNGKey(0)
    sample = model.sample(key, num_samples=num_samples)

    assert sample.mu.shape == (num_samples,)
    assert sample.sigma.shape == (num_samples,)
    assert sample.groups.shape == (num_samples, num_groups)
    assert sample.y_obs.shape == (num_samples, num_groups)


def test_sample_forward_hierarchical_regression():
    num_groups = 5
    num_samples = 10

    def slope_prior(mu_slope, sigma_slope):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_slope), sigma_slope),
            reinterpreted_batch_ndims=1,
        )

    def intercept_prior(mu_intercept, sigma_intercept):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_intercept), sigma_intercept),
            reinterpreted_batch_ndims=1,
        )

    mu_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_slope")
    mu_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_intercept")
    sigma_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma_slope")
    sigma_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="sigma_intercept")

    slopes = paz.Latent(slope_prior, name="slopes")(mu_slope, sigma_slope)
    intercepts = paz.Latent(intercept_prior, name="intercepts")(
        mu_intercept, sigma_intercept
    )

    X = jp.linspace(0, 1, 50)
    group_idx = jp.repeat(jp.arange(num_groups), 10)

    def likelihood(slopes, intercepts):
        means = slopes[group_idx] * X + intercepts[group_idx]
        return tfd.Normal(means, 0.1)

    y_obs = paz.Observable(likelihood, name="y_obs")(slopes, intercepts)
    priors = [mu_slope, mu_intercept, sigma_slope, sigma_intercept]
    model = paz.PGM(priors, [y_obs], "hierarchical_regression")

    key = jax.random.PRNGKey(42)
    sample = model.sample(key, num_samples=num_samples)

    assert sample.mu_slope.shape == (num_samples,)
    assert sample.mu_intercept.shape == (num_samples,)
    assert sample.slopes.shape == (num_samples, num_groups)
    assert sample.intercepts.shape == (num_samples, num_groups)
    assert sample.y_obs.shape == (num_samples, 50)


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


def test_prior_log_prob_inverse_is_consistent():
    model, _, _ = build_bijected_prior_model()
    key = jax.random.PRNGKey(0)
    theta_inv = model.sample_inverse(key)
    state = model.prior.log_prob_inverse(theta_inv)
    log_prob = model.prior.log_prob_inverse(theta_inv).log_prob_sum
    assert jp.isclose(log_prob, state.log_prob_sum)


def test_prior_log_prob_fwd_matches_distribution():
    model, bijector, (low, high) = build_bijected_prior_model()
    key = jax.random.PRNGKey(1)
    theta_inv = model.prior.sample_inverse(key)
    theta_fwd = to_forward_samples(model.prior.latent_space, theta_inv)
    expected_fwd = tfd.Uniform(low, high).log_prob(theta_fwd.x)
    actual_fwd = model.prior.log_prob(theta_fwd).log_prob_sum
    actual_inv = model.prior.log_prob_inverse(theta_inv).log_prob_sum
    expected_inv = expected_fwd + bijector.forward_log_det_jacobian(theta_inv.x)
    assert jp.isclose(actual_fwd, expected_fwd)
    assert jp.isclose(actual_inv, expected_inv)


def test_prior_sample_forward_respects_bounds():
    model, _, (low, high) = build_bijected_prior_model()
    key = jax.random.PRNGKey(2)
    samples = model.prior.sample(key, num_samples=1000)
    assert samples.x.min() >= low
    assert samples.x.max() <= high


def test_likelihood_log_prob_matches_observables():
    model, data = build_two_observables_model()
    key = jax.random.PRNGKey(3)
    theta_inv = model.sample_inverse(key)
    state = model.likelihood.log_prob_inverse(theta_inv, data)
    expected = state.log_prob["y1"] + state.log_prob["y2"]
    actual = model.likelihood.log_prob_inverse(theta_inv, data).log_prob_sum
    assert jp.isclose(actual, expected)


def test_likelihood_accepts_forward_samples_and_list_data():
    model, data = build_two_observables_model()
    key = jax.random.PRNGKey(4)
    theta_inv = model.prior.sample_inverse(key)
    theta_fwd = to_forward_samples(model.prior.latent_space, theta_inv)
    data_list = [data["y1"], data["y2"]]
    inv_log_prob = model.likelihood.log_prob_inverse(
        theta_inv, data
    ).log_prob_sum
    fwd_log_prob = model.likelihood.log_prob(
        theta_fwd, data_list
    ).log_prob_sum
    assert jp.isclose(inv_log_prob, fwd_log_prob)


def test_likelihood_accepts_single_output_value():
    model, observations = build_single_observable_model()
    key = jax.random.PRNGKey(5)
    theta_inv = model.sample_inverse(key)
    expected = model.likelihood.log_prob_inverse(
        theta_inv, {"y": observations}
    ).log_prob_sum
    actual = model.likelihood.log_prob_inverse(
        theta_inv, observations
    ).log_prob_sum
    assert jp.isclose(actual, expected)


def test_likelihood_rejects_single_value_for_multi_output():
    model, data = build_two_observables_model()
    key = jax.random.PRNGKey(6)
    theta_inv = model.sample_inverse(key)
    with pytest.raises(TypeError):
        model.likelihood.log_prob_inverse(theta_inv, data["y1"])
