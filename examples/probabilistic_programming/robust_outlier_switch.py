# Robust linear regression with a global outlier switch.
#
# Original model:
#   slope ~ Normal(0, 1)
#   bias  ~ Normal(0, 1)
#   p     ~ Beta(2, 2)
#   z     ~ Bernoulli(p)
#   y_i | slope, bias, z ~ Normal(slope * x_i + bias, sigma_z)
#
# The switch z selects between a tight inlier noise (sigma_in) and a wide
# outlier noise (sigma_out). We marginalize z and run MCMC on slope, bias, p,
# then recover p(z | y) from the samples to show which noise regime wins.

import argparse
from collections import namedtuple

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot

Line = namedtuple("Line", ["slope", "bias"])


def build_switch_model(x, stdvs):
    standard = paz.distributions.Normal(0.0, 1.0)
    slope = paz.Prior(standard, name="slope")
    bias = paz.Prior(standard, name="bias")
    beta = paz.distributions.Beta(2.0, 2.0)
    p = paz.Prior(beta, paz.bijectors.Sigmoid(), "p")
    z = paz.Latent(build_switch, name="z")(p)
    y = paz.Observable(Likelihood(x, stdvs), name="y")(slope, bias, z)
    return paz.PGM([slope, bias, p], [y], "robust_outlier_switch")


def build_switch(p):
    return paz.distributions.Bernoulli(probs=p, dtype=jp.float32)


def Likelihood(x, stdvs):
    def apply(slope, bias, z):
        stdv = jp.where(z == 1.0, stdvs[1], stdvs[0])
        return paz.distributions.Normal(slope * x + bias, stdv)

    return apply


def evaluate(line, x):
    return line.slope * x + line.bias


def plot_lines(axis, x, observations, mean_line, density_line, true_line):
    kwargs = {"color": "tab:gray", "alpha": 0.8, "label": "data"}
    plot.scatter(x, observations, axis, **kwargs)
    mean = evaluate(mean_line, x)
    plot.line(x, mean, axis, "black", label="posterior mean line")
    kwargs = {"linestyle": "--", "label": "gaussian approx"}
    plot.line(x, evaluate(density_line, x), axis, "tab:purple", **kwargs)
    kwargs = {"linestyle": "--", "label": "true line"}
    plot.line(x, evaluate(true_line, x), axis, "tab:green", **kwargs)


def plot_noise_bands(axis, x, mean, stdvs, selected):
    colors = ("tab:blue", "tab:orange")
    band = 2.0 * stdvs[selected]
    label = f"selected band (sigma={stdvs[selected]})"
    band_args = (x, mean - band, mean + band, axis, colors[selected])
    plot.fill_between(*band_args, alpha=0.25, label=label)
    other = 1 - selected
    band = 2.0 * stdvs[other]
    label = f"other band (sigma={stdvs[other]})"
    kwargs = {"color": colors[other], "linestyle": ":", "linewidth": 1.5}
    plot.line(x, mean + band, axis, label=label, **kwargs)
    plot.line(x, mean - band, axis, **kwargs)


def plot_switch_posterior(axis, posterior_z):
    labels = ["z=0 (inlier)", "z=1 (outlier)"]
    colors = ["tab:blue", "tab:orange"]
    axis.bar([0, 1], posterior_z, color=colors, tick_label=labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Posterior probability")
    plot.clean(axis)
    axis.set_title("Posterior over switch z")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression with a switch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_observations", type=int, default=30)
    parser.add_argument("--inlier_stdv", type=float, default=0.2)
    parser.add_argument("--outlier_stdv", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=600)
    parser.add_argument("--num_chains", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.25)
    args = parser.parse_args()

    true_line = Line(slope=1.2, bias=-0.2)
    stdvs = (args.inlier_stdv, args.outlier_stdv)
    x = jp.linspace(-1.0, 1.0, args.num_observations)
    switch_model = build_switch_model(x, stdvs)
    model = paz.marginalize(switch_model, ["z"])
    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(nrows=2, ncols=2, figsize=(14, 8))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), (2, 3))
    for true_z, (noise_key, infer_key, density_key) in enumerate(keys):
        noise = jax.random.normal(noise_key, x.shape)
        observations = evaluate(true_line, x) + stdvs[true_z] * noise
        kwargs = {"num_samples": args.num_samples}
        posterior = model.infer(infer_key, observations, **kwargs)
        samples = jax.tree.map(jp.ravel, posterior.inverse_samples)
        switch_args = (switch_model, "z", samples, observations)
        switch = paz.recover_discrete_posterior(*switch_args)
        posterior_z = switch.posterior.mean(axis=0)
        mean_line = Line(samples.slope.mean(), samples.bias.mean())
        density = posterior.as_density(method="gaussian")
        draw = density.sample(density_key, num_samples=1)
        density_line = Line(draw.slope, draw.bias)

        print(f"true z={true_z}, true line={tuple(true_line)}")
        print(f"  posterior slope={mean_line.slope:.3f}")
        print(f"  posterior bias={mean_line.bias:.3f}")
        print(f"  posterior p(z=1 | y)={posterior_z[1]:.3f}")
        rate = posterior.infos.acceptance_rate.mean()
        print(f"  acceptance rate={rate:.3f}")

        axis = axes[0, true_z]
        lines = (mean_line, density_line, true_line)
        plot_lines(axis, x, observations, *lines)
        selected = int(posterior_z[1] >= posterior_z[0])
        plot_noise_bands(axis, x, evaluate(mean_line, x), stdvs, selected)
        plot.legend(axis, loc="upper left")
        plot.clean(axis)
        axis.set_title(f"true z={true_z}, p(z=1|y)={posterior_z[1]:.2f}")
        plot_switch_posterior(axes[1, true_z], posterior_z)
    figure.tight_layout()
    plot.show()
