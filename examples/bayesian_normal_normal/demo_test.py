import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.types import SampleType

from examples.bayesian_normal_normal import demo

tfd = tfp.distributions


def test_log_prob_matches_manual():
    prior_mean = 0.0
    prior_stdv = 1.0
    likelihood_stdv = 0.5
    observations = jp.array([0.2, -0.1, 0.3])
    num_observations = observations.shape[0]

    model = demo.build_normal_normal_model(
        prior_mean, prior_stdv, likelihood_stdv, num_observations
    )
    data = {"x": observations}

    Sample = SampleType(["mu"])
    mu_value = jp.array(0.25)
    samples = Sample(mu_value)

    log_prob = model.apply(samples, data).log_prob_sum
    manual = tfd.Normal(prior_mean, prior_stdv).log_prob(mu_value)
    manual = manual + tfd.Normal(mu_value, likelihood_stdv).log_prob(
        observations
    ).sum()

    assert jp.allclose(log_prob, manual)
