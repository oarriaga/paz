"""
Gaussian Mixture Model: Exact vs Marginalized vs MCMC Inference

This script is a small narrative example that starts from the original
latent-variable model and then shows how marginalization changes the inference
problem. The goal is to verify that both approaches agree numerically.

Original model (explicit latent):
    p ~ Beta(alpha, beta)
    z ~ Bernoulli(p)
    y_i | z ~ Normal(mu_z, sigma)

This model has a discrete latent z. Exact inference over p can be done by
enumerating z in the joint:
    log p(y, p) = logsumexp_z log p(y, z, p)

Marginalized model (z integrated out):
    p ~ Beta(alpha, beta)
    y_i ~ mixture( Normal(mu0, sigma), Normal(mu1, sigma), weights=[1-p, p] )

In code, the marginalization wrapper performs the same log-sum-exp over z.
We use it as a drop-in replacement to compute log p(y, p) without keeping z in
the parameter state.

We compare three routes:
1) Exact enumeration over z (on a grid in p).
2) Automatic marginalization with paz.marginalize(["z"]).
3) Metropolis-Hastings sampling over p using the marginalized log density.

Posterior recovery for z uses:
    p(z | y, p) = softmax_z log p(y, z, p)
and averages over p samples to approximate p(z | y).

The plots show:
    - Observations and component means.
    - log p(y, p, z=0/1) and log p(y, p) from enumeration and marginalization.
    - Posterior p: exact curve vs MCMC histogram.
    - Posterior z: bar plot for enum, marginalization, and MCMC.

The printed diagnostics report max absolute differences between exact and
approximations, verifying consistency.
"""

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.types import SampleType

tfd = tfp.distributions


def build_mixture_model(prior_p, mean0, mean1, likelihood_stdv):
    p = paz.Prior(prior_p, name="p")

    def z_distribution(p):
        return tfd.Bernoulli(probs=p, dtype=jp.float32)

    z = paz.Latent(z_distribution, name="z")(p)

    def y_distribution(z):
        mean = jp.where(z == 1.0, mean1, mean0)
        return tfd.Normal(mean, likelihood_stdv)

    y_obs = paz.Observable(y_distribution, name="y")(z)
    return paz.PGM([p], [y_obs], "gaussian_mixture")


def compute_log_joint_over_grid(
    prior_p, mean0, mean1, likelihood_stdv, observations, p_grid
):
    log_prior = prior_p.log_prob(p_grid)
    bernoulli = tfd.Bernoulli(probs=p_grid, dtype=jp.float32)
    log_prob_z0 = bernoulli.log_prob(jp.array(0.0))
    log_prob_z1 = bernoulli.log_prob(jp.array(1.0))
    log_likelihood_z0 = tfd.Normal(mean0, likelihood_stdv).log_prob(
        observations
    ).sum()
    log_likelihood_z1 = tfd.Normal(mean1, likelihood_stdv).log_prob(
        observations
    ).sum()
    log_joint_z0 = log_prior + log_prob_z0 + log_likelihood_z0
    log_joint_z1 = log_prior + log_prob_z1 + log_likelihood_z1
    return log_joint_z0, log_joint_z1


def compute_log_marginal_pgm(model_marg, data, p_grid):
    Theta = SampleType(["p"])

    def log_prob_for_p(p_value):
        log_prior = model_marg.prior.log_prob(Theta(p_value), space="inv")
        log_like = model_marg.likelihood.log_prob(
            Theta(p_value), data, space="inv"
        )
        return log_prior + log_like

    return jax.vmap(log_prob_for_p)(p_grid)


def normalize_log_density(log_density, grid):
    stabilized = log_density - jp.max(log_density)
    density = jp.exp(stabilized)
    delta = grid[1] - grid[0]
    normalization = density.sum() * delta
    return density / normalization


def compute_posterior_z_from_grid(log_joint_z0, log_joint_z1):
    joint_z0 = jp.exp(log_joint_z0)
    joint_z1 = jp.exp(log_joint_z1)
    normalization = joint_z0.sum() + joint_z1.sum()
    return jp.array([joint_z0.sum(), joint_z1.sum()]) / normalization


def compute_posterior_z_from_marginal(model_marg, data, p_grid, posterior_p):
    Theta = SampleType(["p"])
    posterior_z_given_p = paz.recover_discrete_posterior(
        model_marg, "z", Theta(p_grid), data
    ).posterior
    posterior_z = (posterior_p[:, None] * posterior_z_given_p).sum(axis=0)
    return posterior_z / posterior_z.sum()


def plot_results(
    observations,
    mean0,
    mean1,
    p_grid,
    log_joint_z0,
    log_joint_z1,
    log_marginal_enum,
    log_marginal_pgm,
    posterior_p_enum,
    posterior_p_marg,
    posterior_z_enum,
    posterior_z_marg,
    p_samples,
    posterior_z_mcmc,
    posterior_p_density=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(observations, bins=20, density=True, alpha=0.7)
    axes[0, 0].axvline(mean0, color="tab:blue", linestyle="--", label="mean0")
    axes[0, 0].axvline(mean1, color="tab:orange", linestyle="--", label="mean1")
    axes[0, 0].set_title("Observations and component means")
    axes[0, 0].legend()

    axes[0, 1].plot(p_grid, log_joint_z0, label="log p(y,p,z=0)")
    axes[0, 1].plot(p_grid, log_joint_z1, label="log p(y,p,z=1)")
    axes[0, 1].plot(p_grid, log_marginal_enum, "--", label="log p(y,p) enum")
    axes[0, 1].plot(p_grid, log_marginal_pgm, ":", label="log p(y,p) marg")
    axes[0, 1].set_title("Joint and marginalized log densities")
    axes[0, 1].legend()

    axes[1, 0].hist(
        p_samples,
        bins=30,
        density=True,
        alpha=0.4,
        label="posterior p density (MCMC)",
    )
    axes[1, 0].plot(
        p_grid, posterior_p_enum, label="posterior p density (enum)"
    )
    axes[1, 0].plot(
        p_grid, posterior_p_marg, "--", label="posterior p density (marg)"
    )
    if posterior_p_density is not None:
        axes[1, 0].plot(
            p_grid,
            posterior_p_density,
            ":",
            label="posterior p density (gaussian fit)",
        )
    axes[1, 0].set_title("Posterior over p")
    axes[1, 0].legend()

    bar_positions = jp.arange(2)
    bar_width = 0.35
    axes[1, 1].bar(
        bar_positions - bar_width,
        posterior_z_enum,
        width=bar_width,
        label="enum",
    )
    axes[1, 1].bar(
        bar_positions,
        posterior_z_marg,
        width=bar_width,
        label="marg",
    )
    axes[1, 1].bar(
        bar_positions + bar_width,
        posterior_z_mcmc,
        width=bar_width,
        label="mcmc",
    )
    axes[1, 1].set_xticks(bar_positions)
    axes[1, 1].set_xticklabels(["z=0", "z=1"])
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Posterior over z")
    axes[1, 1].legend()

    plt.tight_layout()
    return fig


def main():
    key = jax.random.PRNGKey(0)
    num_observations = 8
    mean0 = -0.2
    mean1 = 0.2
    likelihood_stdv = 1.0
    true_p = 0.4
    prior_p = tfd.Beta(2.0, 2.0)

    key, z_key, y_key = jax.random.split(key, 3)
    z_true = jax.random.bernoulli(z_key, true_p)
    component_mean = mean1 if z_true else mean0
    observations = component_mean + likelihood_stdv * jax.random.normal(
        y_key, (num_observations,)
    )

    model = build_mixture_model(prior_p, mean0, mean1, likelihood_stdv)
    data = {"y": observations}
    model_marg = paz.marginalize(model, ["z"])

    p_grid = jp.linspace(0.01, 0.99, 200)
    log_joint_z0, log_joint_z1 = compute_log_joint_over_grid(
        prior_p, mean0, mean1, likelihood_stdv, observations, p_grid
    )
    log_marginal_enum = logsumexp(
        jp.stack([log_joint_z0, log_joint_z1]), axis=0
    )
    log_marginal_pgm = compute_log_marginal_pgm(model_marg, data, p_grid)

    posterior_p_enum = normalize_log_density(log_marginal_enum, p_grid)
    posterior_p_marg = normalize_log_density(log_marginal_pgm, p_grid)

    posterior_z_enum = compute_posterior_z_from_grid(
        log_joint_z0, log_joint_z1
    )
    posterior_z_marg = compute_posterior_z_from_marginal(
        model_marg, data, p_grid, posterior_p_marg
    )

    num_samples = 1200
    num_chains = 2
    burn_in = 300
    sigma = 0.2

    key, tune_key, infer_key = jax.random.split(key, 3)
    model_marg.tune(
        tune_key,
        data,
        num_chains=num_chains,
        sigma=sigma,
        warmup=burn_in,
        num_samples=num_samples,
    )
    posterior = model_marg.infer(infer_key, data, num_samples=num_samples)
    p_samples = posterior.samples.position.p.reshape(-1)
    posterior_density = posterior.as_density(method="gaussian")
    Theta = SampleType(["p"])
    posterior_p_density = posterior_density.prob(Theta(p_grid))
    mcmc_mean = p_samples.mean()
    mcmc_stdv = p_samples.std()

    delta = p_grid[1] - p_grid[0]
    exact_mean = (posterior_p_enum * p_grid).sum() * delta
    exact_var = (posterior_p_enum * (p_grid - exact_mean) ** 2).sum() * delta
    exact_stdv = jp.sqrt(exact_var)

    posterior_z_given_p = paz.recover_discrete_posterior(
        model_marg, "z", Theta(p_samples), data
    ).posterior
    posterior_z_mcmc = posterior_z_given_p.mean(axis=0)
    posterior_z_mcmc = posterior_z_mcmc / posterior_z_mcmc.sum()

    max_log_diff = jp.max(jp.abs(log_marginal_enum - log_marginal_pgm))
    max_p_diff = jp.max(jp.abs(posterior_p_enum - posterior_p_marg))
    max_z_diff = jp.max(jp.abs(posterior_z_enum - posterior_z_marg))
    max_z_mcmc_diff = jp.max(jp.abs(posterior_z_enum - posterior_z_mcmc))

    print("=" * 60)
    print("Gaussian mixture inference comparison")
    print("=" * 60)
    print(f"true_p={true_p}, z_true={int(z_true)}")
    print(f"max log p(y,p) diff: {max_log_diff:.6e}")
    print(f"max posterior p diff: {max_p_diff:.6e}")
    print(f"max posterior z diff: {max_z_diff:.6e}")
    print(f"max posterior z diff (mcmc): {max_z_mcmc_diff:.6e}")
    print(f"posterior p mean (exact): {exact_mean:.6f}")
    print(f"posterior p mean (mcmc): {mcmc_mean:.6f}")
    print(f"posterior p stdv (exact): {exact_stdv:.6f}")
    print(f"posterior p stdv (mcmc): {mcmc_stdv:.6f}")
    print(f"posterior z (enum): {posterior_z_enum}")
    print(f"posterior z (marg): {posterior_z_marg}")
    print(f"posterior z (mcmc): {posterior_z_mcmc}")

    plot_results(
        observations,
        mean0,
        mean1,
        p_grid,
        log_joint_z0,
        log_joint_z1,
        log_marginal_enum,
        log_marginal_pgm,
        posterior_p_enum,
        posterior_p_marg,
        posterior_z_enum,
        posterior_z_marg,
        p_samples,
        posterior_z_mcmc,
        posterior_p_density,
    )
    plt.show()


if __name__ == "__main__":
    main()
