import jax.numpy as jp
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

import paz

from examples.gaussian_mixture_model import inference_comparison as example

tfd = tfp.distributions


def test_posterior_z_matches_enum_on_grid():
    prior_p = tfd.Beta(2.0, 2.0)
    mean0 = -0.2
    mean1 = 0.2
    likelihood_stdv = 1.0
    observations = jp.array([0.1, -0.4, 0.3])

    model = example.build_mixture_model(prior_p, mean0, mean1, likelihood_stdv)
    data = {"y": observations}
    model_marg = paz.marginalize(model, ["z"])

    p_grid = jp.linspace(0.1, 0.9, 5)
    log_joint_z0, log_joint_z1 = example.compute_log_joint_over_grid(
        prior_p, mean0, mean1, likelihood_stdv, observations, p_grid
    )
    log_marginal_enum = logsumexp(
        jp.stack([log_joint_z0, log_joint_z1]), axis=0
    )
    posterior_p = example.normalize_log_density(log_marginal_enum, p_grid)
    posterior_z_enum = example.compute_posterior_z_from_grid(
        log_joint_z0, log_joint_z1
    )
    posterior_z_marg = example.compute_posterior_z_from_marginal(
        model_marg, data, p_grid, posterior_p
    )

    assert jp.allclose(posterior_z_enum, posterior_z_marg)
