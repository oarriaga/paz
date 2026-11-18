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

import argparse

import jax
import jax.numpy as jp
from jax.scipy.special import logsumexp

import paz
import paz.utils.plot as plot
from paz.inference.types import SampleType


def build_mixture_model(prior_p, means, stdv):
    p = paz.Prior(prior_p, name="p")
    z = paz.Latent(build_switch, name="z")(p)
    y = paz.Observable(Likelihood(means, stdv), name="y")(z)
    return paz.PGM([p], [y], "gaussian_mixture")


def build_switch(p):
    return paz.distributions.Bernoulli(probs=p, dtype=jp.float32)


def Likelihood(means, stdv):
    def apply(z):
        mean = jp.where(z == 1.0, means[1], means[0])
        return paz.distributions.Normal(mean, stdv)

    return apply


def compute_log_joints(prior_p, means, stdv, observations, p_grid):
    switch = build_switch(p_grid)
    log_joints = []
    for z, mean in enumerate(means):
        normal = paz.distributions.Normal(mean, stdv)
        log_prior = prior_p.log_prob(p_grid) + switch.log_prob(float(z))
        log_joints.append(log_prior + normal.log_prob(observations).sum())
    return jp.stack(log_joints)


def normalize_density(log_density, grid):
    density = jp.exp(log_density - log_density.max())
    return density / (density.sum() * (grid[1] - grid[0]))


def compute_grid_moments(density, grid):
    delta = grid[1] - grid[0]
    mean = (density * grid).sum() * delta
    variance = (density * (grid - mean) ** 2).sum() * delta
    return mean, jp.sqrt(variance)


def average_switch_posterior(model, data, p_values, weights):
    samples = SampleType(["p"])(p_values)
    switch = paz.recover_discrete_posterior(model, "z", samples, data)
    posterior_z = (weights[:, None] * switch.posterior).sum(axis=0)
    return posterior_z / posterior_z.sum()


def plot_observations(axis, observations, means):
    plot.histogram(observations, axis, bins=20)
    plot.vline(means[0], axis, color="tab:blue", label="mean 0")
    plot.vline(means[1], axis, color="tab:orange", label="mean 1")
    plot.legend(axis)
    plot.clean(axis)
    axis.set_title("Observations and component means")


def plot_log_densities(axis, p_grid, log_joints, log_enum, log_marginal):
    plot.line(p_grid, log_joints[0], axis, "C0", label="log p(y,p,z=0)")
    plot.line(p_grid, log_joints[1], axis, "C1", label="log p(y,p,z=1)")
    kwargs = {"linestyle": "--", "label": "log p(y,p) enum"}
    plot.line(p_grid, log_enum, axis, "C2", **kwargs)
    kwargs = {"linestyle": ":", "label": "log p(y,p) marg"}
    plot.line(p_grid, log_marginal, axis, "C3", **kwargs)
    plot.legend(axis)
    plot.clean(axis)
    axis.set_title("Joint and marginalized log densities")


def plot_posterior_p(axis, p_grid, p_samples, densities):
    plot.histogram(p_samples, axis, bins=30, alpha=0.4, label="MCMC")
    labels = ["enum", "marg", "gaussian fit"]
    plot.compare_densities(p_grid, densities, labels, axis)
    plot.legend(axis)
    plot.clean(axis)
    axis.set_title("Posterior over p")


def plot_posterior_z(axis, posteriors, labels, width=0.35):
    positions = jp.arange(2)
    for offset, posterior, label in zip((-1, 0, 1), posteriors, labels):
        x = positions + offset * width
        color = f"C{offset + 1}"
        plot.bar(x, posterior, axis, color, width=width, label=label)
    axis.set_xticks(positions)
    axis.set_xticklabels(["z=0", "z=1"])
    axis.set_ylim(0.0, 1.0)
    plot.legend(axis)
    plot.clean(axis)
    axis.set_title("Posterior over z")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mixture inference comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_observations", type=int, default=8)
    parser.add_argument("--means", type=float, nargs=2, default=(-0.2, 0.2))
    parser.add_argument("--stdv", type=float, default=1.0)
    parser.add_argument("--true_p", type=float, default=0.4)
    parser.add_argument("--num_samples", type=int, default=1200)
    parser.add_argument("--num_chains", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.2)
    args = parser.parse_args()

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    true_z = int(jax.random.bernoulli(keys[0], args.true_p))
    noise = jax.random.normal(keys[1], (args.num_observations,))
    observations = args.means[true_z] + args.stdv * noise
    prior_p = paz.distributions.Beta(2.0, 2.0)
    mixture_model = build_mixture_model(prior_p, args.means, args.stdv)
    model = paz.marginalize(mixture_model, ["z"])

    p_grid = jp.linspace(0.01, 0.99, 200)
    model_args = (prior_p, args.means, args.stdv, observations)
    log_joints = compute_log_joints(*model_args, p_grid)
    log_enum = logsumexp(log_joints, axis=0)
    grid_samples = SampleType(["p"])(p_grid)
    log_marginal = model.log_prob_inverse(grid_samples, observations)
    p_enum = normalize_density(log_enum, p_grid)
    p_marginal = normalize_density(log_marginal, p_grid)
    z_enum = jax.nn.softmax(logsumexp(log_joints, axis=1))
    switch_args = (mixture_model, observations, p_grid, p_marginal)
    z_marginal = average_switch_posterior(*switch_args)

    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    posterior = model.infer(keys[2], observations, num_samples=args.num_samples)
    p_samples = posterior.samples.p.flatten()
    weights = jp.ones_like(p_samples)
    switch_args = (mixture_model, observations, p_samples, weights)
    z_mcmc = average_switch_posterior(*switch_args)
    p_gaussian = posterior.as_density(method="gaussian").prob(grid_samples)
    exact_mean, exact_stdv = compute_grid_moments(p_enum, p_grid)

    print(f"true_p={args.true_p}, true_z={true_z}")
    print(f"max log p(y,p) diff: {jp.abs(log_enum - log_marginal).max():.6e}")
    print(f"max posterior p diff: {jp.abs(p_enum - p_marginal).max():.6e}")
    print(f"max posterior z diff: {jp.abs(z_enum - z_marginal).max():.6e}")
    print(f"max posterior z diff (mcmc): {jp.abs(z_enum - z_mcmc).max():.6e}")
    print(f"posterior p mean (exact): {exact_mean:.6f}")
    print(f"posterior p mean (mcmc): {p_samples.mean():.6f}")
    print(f"posterior p stdv (exact): {exact_stdv:.6f}")
    print(f"posterior p stdv (mcmc): {p_samples.std():.6f}")
    print(f"posterior z (enum): {z_enum}")
    print(f"posterior z (marg): {z_marginal}")
    print(f"posterior z (mcmc): {z_mcmc}")

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(nrows=2, ncols=2, figsize=(12, 8))
    plot_observations(axes[0, 0], observations, args.means)
    log_densities = (log_joints, log_enum, log_marginal)
    plot_log_densities(axes[0, 1], p_grid, *log_densities)
    densities = [p_enum, p_marginal, p_gaussian]
    plot_posterior_p(axes[1, 0], p_grid, p_samples, densities)
    posteriors, labels = (z_enum, z_marginal, z_mcmc), ("enum", "marg", "mcmc")
    plot_posterior_z(axes[1, 1], posteriors, labels)
    figure.tight_layout()
    plot.show()
