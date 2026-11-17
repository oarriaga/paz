"""
Normal-Normal Bayesian Model with Analytical Posterior Verification

Model:
    Prior:      mu ~ Normal(mu_0, sigma_0)
    Likelihood: X_1, ..., X_n | mu ~ Normal(mu, sigma)  [sigma known]

Analytical Posterior:
    mu | X ~ Normal(mu_post, sigma_post)

    where:
        sigma^2_post = 1 / (1/sigma_0^2 + n/sigma^2)
        mu_post  = sigma^2_post * (mu_0/sigma_0^2 + n*X_bar/sigma^2)

This example verifies that MCMC samples converge to the analytical posterior.
"""

import argparse
from collections import namedtuple

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot

AnalyticalPosterior = namedtuple("AnalyticalPosterior", ["mean", "stdv"])


def compute_analytical_posterior(prior_mean, prior_stdv, stdv, observations):
    prior_precision = 1.0 / prior_stdv**2
    data_precision = len(observations) / stdv**2
    variance = 1.0 / (prior_precision + data_precision)
    weighted_prior = prior_mean * prior_precision
    weighted_data = observations.mean() * data_precision
    mean = variance * (weighted_prior + weighted_data)
    return AnalyticalPosterior(mean, jp.sqrt(variance))


def build_model(prior_mean, prior_stdv, stdv, num_observations):
    normal = paz.distributions.Normal(prior_mean, prior_stdv)
    mu = paz.Prior(normal, name="mu")
    likelihood = Likelihood(stdv, num_observations)
    x = paz.Observable(likelihood, name="x")(mu)
    return paz.PGM([mu], [x], "normal_normal")


def Likelihood(stdv, num_observations):
    def apply(mu):
        return paz.distributions.Normal(jp.full(num_observations, mu), stdv)

    return apply


def compute_relative_error(estimate, reference):
    return abs(estimate - reference) / abs(reference)


def describe_check(error, tolerance):
    return f"{error:.2%} {'[OK]' if error < tolerance else '[FAIL]'}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal-Normal model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior_mean", type=float, default=0.0)
    parser.add_argument("--prior_stdv", type=float, default=2.0)
    parser.add_argument("--stdv", type=float, default=1.0)
    parser.add_argument("--true_mu", type=float, default=3.0)
    parser.add_argument("--num_observations", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_chains", type=int, default=4)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--mean_tolerance", type=float, default=0.05)
    parser.add_argument("--stdv_tolerance", type=float, default=0.1)
    args = parser.parse_args()

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    noise = jax.random.normal(keys[0], (args.num_observations,))
    observations = args.true_mu + args.stdv * noise
    prior_args = (args.prior_mean, args.prior_stdv, args.stdv)
    analytical = compute_analytical_posterior(*prior_args, observations)
    print(f"Prior:      mu ~ Normal({args.prior_mean}, {args.prior_stdv})")
    print(f"Likelihood: X | mu ~ Normal(mu, {args.stdv})")
    print(f"True mu: {args.true_mu}, data mean: {observations.mean():.4f}")
    print(f"Analytical: Normal({analytical.mean:.4f}, {analytical.stdv:.4f})")

    model = build_model(*prior_args, args.num_observations)
    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    posterior = model.infer(keys[1], observations, num_samples=args.num_samples)
    acceptance_rates = posterior.infos.acceptance_rate
    chains = posterior.samples.mu
    draws = chains.flatten()
    mean_error = compute_relative_error(draws.mean(), analytical.mean)
    stdv_error = compute_relative_error(draws.std(), analytical.stdv)
    print(f"Acceptance rate: {acceptance_rates.mean():.3f}")
    print(f"MCMC:       Normal({draws.mean():.4f}, {draws.std():.4f})")
    print(f"Mean error: {describe_check(mean_error, args.mean_tolerance)}")
    print(f"Stdv error: {describe_check(stdv_error, args.stdv_tolerance)}")

    plot.configure(fontsize=14)
    plot.trace_panel({"mu": chains}, title="MCMC Traces")
    plot.show()

    figure, axis = plot.subplots()
    plot.trace_lines(chains, axis)
    kwargs = {"color": plot.EARTH.primary, "label": "Analytical mean"}
    plot.hline(float(analytical.mean), axis, **kwargs)
    kwargs = {"color": plot.EARTH.secondary, "linestyle": ":"}
    plot.hline(args.true_mu, axis, label="True mu", **kwargs)
    plot.set_labels(axis, x="Iteration", y="mu")
    plot.clean(axis)
    axis.legend()
    axis.set_title("Trace plot (all chains)")
    plot.show()

    span = 4 * analytical.stdv
    grid = jp.linspace(analytical.mean - span, analytical.mean + span, 200)
    prior = paz.distributions.Normal(args.prior_mean, args.prior_stdv)
    exact = paz.distributions.Normal(analytical.mean, analytical.stdv)
    gaussian_fit = posterior.as_density(method="gaussian")
    densities = [exact.prob(grid), prior.prob(grid)]
    densities = densities + [gaussian_fit.prob({"mu": grid})]
    labels = ["Analytical posterior", "Prior", "Gaussian fit"]
    colors = [plot.EARTH.primary, plot.EARTH.secondary, "black"]
    figure, axis = plot.subplots()
    plot.histogram(draws, axis, alpha=0.5, color=plot.BLUE_GREY.primary)
    plot.compare_densities(grid, densities, labels, axis, colors)
    plot.set_labels(axis, x="mu", y="Density")
    plot.clean(axis)
    axis.legend()
    axis.set_title("Posterior distribution comparison")
    plot.show()

    prior_draws = prior.sample(len(draws), seed=keys[2])
    plot.prior_posterior_comparison(prior_draws, draws, "mu")
    plot.show()

    figure, axis = plot.subplots()
    qq_args = (draws, analytical.mean, analytical.stdv, axis)
    plot.qq_plot(*qq_args, plot.BLUE_GREY.primary)
    plot.clean(axis)
    axis.set_title("Q-Q plot (vs analytical posterior)")
    plot.show()

    figure, axis = plot.subplots()
    plot.diagnostics(acceptance_rates, axis, color=plot.DANDELION.primary)
    axis.set_title("Acceptance rates per chain")
    plot.show()
