import jax.numpy as jp

import paz
from paz.inference.types import SampleType

from examples.bayesian_linear_regression import robust_outlier_switch


def test_outlier_posterior_normalizes():
    x = jp.linspace(-1.0, 1.0, 3)
    sigma_in = 0.2
    sigma_out = 1.0
    observations = jp.array([0.1, 0.0, -0.1])

    model = robust_outlier_switch.build_switch_model(x, sigma_in, sigma_out)
    model_marg = paz.marginalize(model, ["z"])
    data = {"y": observations}

    Theta = SampleType(["slope", "bias", "p"])
    theta_samples = Theta(0.1, 0.0, 0.0)
    posterior = paz.recover_discrete_posterior(
        model_marg, "z", theta_samples, data
    ).posterior

    assert jp.allclose(posterior.sum(), 1.0)
