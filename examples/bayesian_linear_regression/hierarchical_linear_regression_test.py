import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from examples.bayesian_linear_regression import hierarchical_linear_regression

tfd = tfp.distributions
tfb = tfp.bijectors


def test_log_prob_is_finite():
    true_params = hierarchical_linear_regression.TrueParams(
        mu_slope=0.8,
        sigma_slope=0.3,
        mu_intercept=0.2,
        sigma_intercept=0.2,
        sigma_obs=0.1,
    )
    num_groups = 2
    num_per_group = 3
    key = jax.random.PRNGKey(0)

    X, y, group_idx, _, _ = hierarchical_linear_regression.generate_hierarchical_data(
        key, num_groups, num_per_group, true_params
    )
    sigma_bijector = tfb.Chain(
        [tfb.Shift(0.01), tfb.Scale(0.99), tfb.Sigmoid()]
    )
    obs_bijector = tfb.Chain(
        [tfb.Shift(0.01), tfb.Scale(0.49), tfb.Sigmoid()]
    )

    model = hierarchical_linear_regression.build_centered_model(
        X, group_idx, num_groups, sigma_bijector, obs_bijector
    )
    data = {"y_obs": y}

    sample_key = jax.random.PRNGKey(1)
    samples = model.sample_inverse(sample_key, 1)
    log_prob = model.apply(samples, data).log_prob_sum

    assert jp.isfinite(log_prob)
