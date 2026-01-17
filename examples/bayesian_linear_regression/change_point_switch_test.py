import jax.numpy as jp

import paz
from paz.inference.types import SampleType

from examples.bayesian_linear_regression import change_point_switch


def test_switch_posterior_normalizes():
    x = jp.linspace(-1.0, 1.0, 3)
    sigma = 0.2
    observations = jp.array([0.1, 0.0, -0.1])

    model = change_point_switch.build_switch_model(x, sigma)
    model_marg = paz.marginalize(model, ["switch_index"])
    data = {"y": observations}

    Theta = SampleType(
        ["slope_left", "bias_left", "slope_right", "bias_right"]
    )
    theta_samples = Theta(0.1, 0.0, -0.1, 0.0)
    posterior = paz.recover_discrete_posterior(
        model_marg, "switch_index", theta_samples, data
    ).posterior

    assert jp.allclose(posterior.sum(), 1.0)
