import jax.numpy as jp

import paz
from paz.inference.types import SampleType

from examples.bayesian_linear_regression import multi_change_point_switch


def test_switch_config_posterior_normalizes():
    num_observations = 3
    num_switches = 1
    x = jp.linspace(-1.0, 1.0, num_observations)
    sigma = 0.2
    observations = jp.array([0.1, 0.0, -0.1])

    switch_table = multi_change_point_switch.build_switch_table(
        num_observations, num_switches
    )
    model = multi_change_point_switch.build_switch_model(
        x, sigma, switch_table, num_switches
    )
    model_marg = paz.marginalize(model, ["switch_index"])
    data = {"y": observations}

    Theta = SampleType(
        [
            "slope_segment_0",
            "slope_segment_1",
            "bias_segment_0",
            "bias_segment_1",
        ]
    )
    theta_samples = Theta(0.1, -0.1, 0.0, 0.0)
    posterior = paz.recover_discrete_posterior(
        model_marg, "switch_index", theta_samples, data
    ).posterior

    assert jp.allclose(posterior.sum(), 1.0)
