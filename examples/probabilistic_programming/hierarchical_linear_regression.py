"""
Hierarchical Bayesian Linear Regression

Demonstrates hierarchical modeling with group-level varying slopes and
intercepts. Choose the model with `--parameterization`.

CENTERED PARAMETERIZATION (standard):
  slopes[g] ~ Normal(mu_slope, sigma_slope)
  intercepts[g] ~ Normal(mu_intercept, sigma_intercept)

  Good for: strongly informed groups (large n per group)
  Risk: Neal's funnel pathology when sigma is small

NON-CENTERED PARAMETERIZATION (reparameterized):
  z_slopes[g] ~ Normal(0, 1)
  slopes[g] = mu_slope + sigma_slope * z_slopes[g]

  z_intercepts[g] ~ Normal(0, 1)
  intercepts[g] = mu_intercept + sigma_intercept * z_intercepts[g]

  Good for: weakly informed groups (small n per group, small sigma)
  Avoids: Neal's funnel by decorrelating sigma from group effects

Reference: https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
"""

import argparse
from collections import namedtuple

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot

fields = ["mu_slope", "mu_intercept", "sigma_slope", "sigma_intercept"]
Hyperparameters = namedtuple("Hyperparameters", fields + ["sigma_obs"])
fields = Hyperparameters._fields + ("z_slopes", "z_intercepts")
Noncentered = namedtuple("Noncentered", fields)


def sample_data(key, num_groups, num_per_group, truth):
    keys = jax.random.split(key, 4)
    shape = (num_groups, num_per_group)
    slope_noise = jax.random.normal(keys[0], (num_groups,))
    intercept_noise = jax.random.normal(keys[1], (num_groups,))
    slopes = truth.mu_slope + truth.sigma_slope * slope_noise
    intercepts = truth.mu_intercept + truth.sigma_intercept * intercept_noise
    X = jax.random.uniform(keys[2], shape)
    noise = truth.sigma_obs * jax.random.normal(keys[3], shape)
    y = slopes[:, None] * X + intercepts[:, None] + noise
    groups = jp.repeat(jp.arange(num_groups), num_per_group)
    return X.ravel(), y.ravel(), groups, slopes, intercepts


def build_centered_model(X, groups, num_groups):
    priors = build_hyperpriors()
    mu_slope, mu_intercept, sigma_slope, sigma_intercept, sigma_obs = priors
    slopes = paz.Latent(GroupPrior(num_groups), name="slopes")
    slopes = slopes(mu_slope, sigma_slope)
    intercepts = paz.Latent(GroupPrior(num_groups), name="intercepts")
    intercepts = intercepts(mu_intercept, sigma_intercept)
    y = paz.Observable(Likelihood(X, groups), name="y_obs")
    y = y(slopes, intercepts, sigma_obs)
    return paz.PGM(priors, [y], "hierarchical_centered")


def build_hyperpriors():
    standard = paz.distributions.Normal(0.0, 1.0)
    mu_slope = paz.Prior(standard, name="mu_slope")
    mu_intercept = paz.Prior(standard, name="mu_intercept")
    sigma_slope = build_bounded_prior("sigma_slope", 0.01, 1.0)
    sigma_intercept = build_bounded_prior("sigma_intercept", 0.01, 1.0)
    sigma_obs = build_bounded_prior("sigma_obs", 0.01, 0.5)
    return [mu_slope, mu_intercept, sigma_slope, sigma_intercept, sigma_obs]


def build_bounded_prior(name, low, high):
    affine = [paz.bijectors.Shift(low), paz.bijectors.Scale(high - low)]
    bijector = paz.bijectors.Chain(affine + [paz.bijectors.Sigmoid()])
    return paz.Prior(paz.distributions.Uniform(low, high), bijector, name)


def GroupPrior(num_groups):
    def apply(mean, stdv):
        normal = paz.distributions.Normal(jp.full(num_groups, mean), stdv)
        return paz.distributions.Independent(normal, 1)

    return apply


def Likelihood(X, groups):
    def apply(slopes, intercepts, sigma_obs):
        means = slopes[groups] * X + intercepts[groups]
        return paz.distributions.Normal(means, sigma_obs)

    return apply


def unpack_group_lines(sample):
    return sample.slopes, sample.intercepts


def build_noncentered_model(X, groups, num_groups):
    normal = paz.distributions.Normal(jp.zeros(num_groups), 1.0)
    offsets = paz.distributions.Independent(normal, 1)
    z_slopes = paz.Prior(offsets, name="z_slopes")
    z_intercepts = paz.Prior(offsets, name="z_intercepts")
    priors = build_hyperpriors() + [z_slopes, z_intercepts]
    likelihood = NoncenteredLikelihood(X, groups)
    y = paz.Observable(likelihood, name="y_obs")(*priors)
    return paz.PGM(priors, [y], "hierarchical_noncentered")


def NoncenteredLikelihood(X, groups):
    likelihood = Likelihood(X, groups)

    def apply(*values):
        sample = Noncentered(*values)
        slopes, intercepts = compute_group_lines(sample)
        return likelihood(slopes, intercepts, sample.sigma_obs)

    return apply


def compute_group_lines(sample):
    slope_offsets = sample.sigma_slope[..., None] * sample.z_slopes
    intercept_offsets = sample.sigma_intercept[..., None] * sample.z_intercepts
    slopes = sample.mu_slope[..., None] + slope_offsets
    intercepts = sample.mu_intercept[..., None] + intercept_offsets
    return slopes, intercepts


def plot_lines(axis, slopes, intercepts, **kwargs):
    grid = jp.linspace(0.0, 1.0, 2)
    axis.plot(grid, slopes * grid[:, None] + intercepts, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical regression")
    choices = ["centered", "non-centered"]
    kwargs = {"default": "centered", "choices": choices}
    parser.add_argument("--parameterization", **kwargs)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_groups", type=int, default=5)
    parser.add_argument("--num_per_group", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=70_000)
    parser.add_argument("--num_chains", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--tune_steps", type=int, default=500)
    parser.add_argument("--tune_episodes", type=int, default=20)
    args = parser.parse_args()

    means = {"mu_slope": 0.8, "mu_intercept": 0.2}
    stdvs = {"sigma_slope": 0.3, "sigma_intercept": 0.2, "sigma_obs": 0.1}
    truth = Hyperparameters(**means, **stdvs)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 5)
    data_args = (args.num_groups, args.num_per_group, truth)
    data = sample_data(keys[0], *data_args)
    X, y, groups, true_slopes, true_intercepts = data
    centered = (build_centered_model, unpack_group_lines)
    noncentered = (build_noncentered_model, compute_group_lines)
    builders = {"centered": centered, "non-centered": noncentered}
    build_model, get_group_lines = builders[args.parameterization]
    model = build_model(X, groups, args.num_groups)
    colors = [f"C{group}" for group in range(args.num_groups)]
    plot.configure(fontsize=12)

    prior_samples = model.sample(keys[1], num_samples=20)
    prior_slopes, prior_intercepts = get_group_lines(prior_samples)
    figure, axes = plot.subplots(ncols=2, figsize=(10, 4))
    for group, color in enumerate(colors):
        lines = (prior_slopes[:, group], prior_intercepts[:, group])
        plot_lines(axes[0], *lines, color=color, alpha=0.1)
    plot.group_scatter(X, y, groups, axes[1], colors)
    for axis, title in zip(axes, ("Prior predictive", "Observed data")):
        axis.set_title(title)
        plot.set_labels(axis, x="X", y="y")
        plot.clean(axis)
    figure.tight_layout()
    plot.show()

    # One step size for all parameters leaves chains stuck at the upper
    # bound of the group stdvs; a step per parameter lets them mix.
    tuner_args = (args.sigma, args.tune_steps, args.tune_episodes)
    tuner = paz.AdaptiveDiagonalTuner(*tuner_args)
    kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    posterior = model.infer(keys[2], y, num_samples=args.num_samples)
    samples = posterior.samples
    acceptance_rates = posterior.infos.acceptance_rate
    slopes, intercepts = get_group_lines(samples)
    slopes = slopes.reshape(-1, args.num_groups)
    intercepts = intercepts.reshape(-1, args.num_groups)
    draw = posterior.as_density(method="gaussian").sample(keys[3])
    density_slopes, density_intercepts = get_group_lines(draw)

    print(f"Parameterization: {args.parameterization}")
    print(f"Mean acceptance rate: {acceptance_rates.mean():.3f}")
    for name in truth._fields:
        estimate = getattr(samples, name).mean()
        print(f"  {name}: {estimate:.3f} (true: {getattr(truth, name)})")
    # Few groups say little about the group stdvs: compare them with the
    # stdv of the sampled group values, not only with the generating value.
    print(f"  stdv of true group slopes: {true_slopes.std(ddof=1):.3f}")
    print(f"  stdv of true group intercepts: {true_intercepts.std(ddof=1):.3f}")
    print("  group slopes:", slopes.mean(axis=0), "true:", true_slopes)
    group_intercepts = intercepts.mean(axis=0)
    print("  group intercepts:", group_intercepts, "true:", true_intercepts)

    draws = {name: getattr(samples, name) for name in truth._fields}
    plot.posterior_panel(draws, truth._asdict())
    plot.show()

    figure, axis = plot.subplots()
    kwargs = {"c": colors, "s": 100}
    axis.scatter(slopes.mean(axis=0), group_intercepts, label="mean", **kwargs)
    true_lines = (true_slopes, true_intercepts)
    axis.scatter(*true_lines, marker="x", label="true", **kwargs)
    kwargs = {"color": "tab:purple", "marker": "D", "label": "gaussian approx"}
    axis.scatter(density_slopes, density_intercepts, s=80, **kwargs)
    plot.set_labels(axis, x="slope", y="intercept")
    plot.legend(axis, loc="upper left")
    plot.clean(axis)
    axis.set_title("Group parameters")
    plot.show()

    figure, axes = plot.subplots(ncols=args.num_groups, figsize=(12, 4))
    indices = jax.random.choice(keys[4], len(slopes), (50,))
    for group, (axis, color) in enumerate(zip(axes, colors)):
        is_group = groups == group
        plot.scatter(X[is_group], y[is_group], axis, 0.6, 20, color)
        lines = (slopes[indices, group], intercepts[indices, group])
        plot_lines(axis, *lines, color=color, alpha=0.1)
        true_line = (true_slopes[group], true_intercepts[group])
        plot_lines(axis, *true_line, color="black", linestyle="--")
        axis.set_title(f"Group {group}")
        plot.set_labels(axis, x="X")
        plot.clean(axis)
    axes[0].set_ylabel("y")
    figure.suptitle("Posterior predictive by group (dashed: true line)")
    figure.tight_layout()
    plot.show()

    figure, axis = plot.subplots()
    plot.diagnostics(acceptance_rates, axis, plot.DANDELION.primary)
    axis.set_title("Acceptance rates per chain")
    plot.show()

    names = ("mu_slope", "mu_intercept")
    corner_draws = {name: draws[name].flatten() for name in names}
    plot.corner(corner_draws, {name: getattr(truth, name) for name in names})
    plot.show()

    figure, axis = plot.subplots(figsize=(8, 5))
    names = [f"slope_{group}" for group in range(args.num_groups)]
    center, spread = slopes.mean(axis=0), slopes.std(axis=0)
    interval = (center - spread, center + spread)
    forest_args = (names, center, *interval, true_slopes, axis)
    plot.forest_plot(*forest_args, plot.BLUE_GREY.primary)
    axis.set_title("Group slope estimates (with 1 stdv)")
    plot.show()
