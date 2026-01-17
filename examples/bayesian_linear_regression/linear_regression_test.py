import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz

from examples.bayesian_linear_regression import linear_regression

tfd = tfp.distributions


def test_log_prob_is_finite():
    X = jp.array([0.0, 1.0])
    observations = jp.array([0.1, 0.9])

    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    bias = paz.Prior(tfd.Normal(0.0, 1.0), name="bias")
    stdv = paz.Prior(tfd.Uniform(0.1, 1.0), name="stdv")
    y = paz.Observable(linear_regression.Likelihood(X), name="y_pred")(
        mean, bias, stdv
    )
    model = paz.PGM([mean, bias, stdv], [y], "line")
    data = {"y_pred": observations}

    key = jax.random.PRNGKey(0)
    samples = model.sample_inverse(key, 1)
    log_prob = model.apply(samples, data).log_prob_sum

    assert jp.isfinite(log_prob)
