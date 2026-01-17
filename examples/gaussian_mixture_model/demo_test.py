import jax.numpy as jp

import paz
from paz.inference.types import SampleType

from examples.gaussian_mixture_model import demo


def test_marginal_log_prob_matches_enum():
    mu0_value = jp.array(0.0)
    mu1_value = jp.array(2.0)
    sigma_value = jp.array(1.0)
    p_value = jp.array(0.3)
    y_value = jp.array(1.5)

    model = demo.build_gaussian_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value
    )
    data = {"y": y_value}

    full_samples_z0 = demo.build_full_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value, jp.array(0.0)
    )
    full_samples_z1 = demo.build_full_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value, jp.array(1.0)
    )
    log_joint_z0 = model.apply(full_samples_z0, data).log_prob_sum
    log_joint_z1 = model.apply(full_samples_z1, data).log_prob_sum
    log_marginal_enum = jp.logaddexp(log_joint_z0, log_joint_z1)

    model_marg = paz.marginalize(model, ["z"])
    theta_samples = demo.build_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    log_marginal = model_marg.apply(theta_samples, data).log_prob_sum

    assert jp.allclose(log_marginal_enum, log_marginal)


def test_posterior_normalizes():
    mu0_value = jp.array(0.0)
    mu1_value = jp.array(2.0)
    sigma_value = jp.array(1.0)
    p_value = jp.array(0.3)
    y_value = jp.array(1.5)

    model = demo.build_gaussian_mixture_model(
        mu0_value, mu1_value, sigma_value, p_value
    )
    data = {"y": y_value}
    model_marg = paz.marginalize(model, ["z"])

    theta_samples = demo.build_theta_inverse_samples(
        mu0_value, mu1_value, sigma_value, p_value
    )
    posterior = paz.recover_discrete_posterior(
        model_marg, "z", theta_samples, data
    ).posterior

    assert jp.allclose(posterior.sum(), 1.0)
