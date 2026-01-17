import jax
import jax.numpy as jp

from examples.bayesian_linear_regression import tune_sigma


def test_log_prob_is_finite():
    inputs = jp.array([0.0, 1.0, 2.0])
    observations = jp.array([0.1, 0.9, 2.1])
    model, _ = tune_sigma.build_model(inputs, 0.01, 0.5)
    data = {"y_pred": observations}

    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key, 1)
    log_prob = model.apply(samples, data).log_prob_sum

    assert jp.isfinite(log_prob)
