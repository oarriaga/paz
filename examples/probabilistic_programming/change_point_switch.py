# Change-point linear regression with a discrete switch index.
#
# Original model:
#   slope_left  ~ Normal(0, 1)
#   bias_left   ~ Normal(0, 1)
#   slope_right ~ Normal(0, 1)
#   bias_right  ~ Normal(0, 1)
#   switch_index ~ Categorical(K)  # K = num_observations - 1
#   y_i | params, switch_index ~ Normal(mean_i, sigma)
#
# The switch_index chooses where the data changes from the left line to the
# right line. We marginalize the discrete switch and infer the continuous
# line parameters with MCMC, then recover p(switch_index | y).

import argparse
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot

Line = namedtuple("Line", ["slope", "bias"])


def build_switch_model(x, stdv):
    standard = paz.distributions.Normal(0.0, 1.0)
    names = ["slope_left", "bias_left", "slope_right", "bias_right"]
    priors = [paz.Prior(standard, name=name) for name in names]
    logits = jp.zeros(len(x) - 1)
    switch = paz.distributions.Categorical(logits=logits, dtype=jp.float32)
    priors.append(paz.Prior(switch, name="switch_index"))
    y = paz.Observable(Likelihood(x, stdv), name="y")(*priors)
    return paz.PGM(priors, [y], "change_point_switch")


def Likelihood(x, stdv):
    def apply(slope_left, bias_left, slope_right, bias_right, switch_index):
        left, right = Line(slope_left, bias_left), Line(slope_right, bias_right)
        is_left = jp.arange(len(x)) <= switch_index.astype(jp.int32)
        mean = compute_piecewise(x, left, right, is_left)
        return paz.distributions.Normal(mean, stdv)

    return apply


def compute_piecewise(x, left, right, is_left):
    return jp.where(is_left, evaluate(left, x), evaluate(right, x))


def evaluate(line, x):
    return line.slope * x + line.bias


def plot_posterior_lines(axis, grid, lines, switch_x, stdv):
    blue = plot.BLUE_GREY.primary
    mean = compute_piecewise(grid, *lines, grid <= switch_x)
    plot.line(grid, mean, axis, "black", label="posterior mean")
    label = f"noise band (sigma={stdv})"
    plot.fill_between(grid, mean - stdv, mean + stdv, axis, blue, 0.15, label)
    kwargs = {"linestyle": ":", "label": "left line (posterior)"}
    plot.line(grid, evaluate(lines[0], grid), axis, blue, **kwargs)
    kwargs = {"linestyle": ":", "label": "right line (posterior)"}
    plot.line(grid, evaluate(lines[1], grid), axis, plot.EARTH.accent, **kwargs)


def plot_switches(axis, map_x, true_x):
    plot.vline(map_x, axis, "black", "-", 2, "posterior switch")
    plot.vline(true_x, axis, plot.EARTH.secondary, label="true switch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change-point regression")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--stdv", type=float, default=0.15)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--num_chains", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.25)
    args = parser.parse_args()

    true_lines = (Line(slope=1.0, bias=0.2), Line(slope=-0.6, bias=0.8))
    true_switch_index = 17
    x = jp.linspace(-1.0, 1.0, 40)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    noise_key, infer_key, density_key = keys
    is_left = jp.arange(len(x)) <= true_switch_index
    noise = args.stdv * jax.random.normal(noise_key, x.shape)
    observations = compute_piecewise(x, *true_lines, is_left) + noise

    switch_model = build_switch_model(x, args.stdv)
    model = paz.marginalize(switch_model, ["switch_index"])
    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    kwargs = {"num_samples": args.num_samples}
    posterior = model.infer(infer_key, observations, **kwargs)
    samples = posterior.inverse_samples
    draws = jax.tree.map(jp.ravel, samples)
    recover = partial(paz.recover_discrete_posterior, switch_model)
    recover = partial(recover, "switch_index", data=observations)
    switch = jax.vmap(recover)(draws)
    switch_posterior = switch.posterior.mean(axis=0)
    map_index = int(jp.argmax(switch_posterior))
    switch_positions = 0.5 * (x[:-1] + x[1:])
    map_x = switch_positions[map_index]
    true_x = switch_positions[true_switch_index]
    acceptance_rates = posterior.infos.acceptance_rate
    print(f"true switch index={true_switch_index}, x={true_x:.3f}")
    print(f"posterior switch index={map_index}, x={map_x:.3f}")
    print(f"acceptance rate={acceptance_rates.mean():.3f}")

    means = jax.tree.map(jp.mean, draws)
    left = Line(means.slope_left, means.bias_left)
    right = Line(means.slope_right, means.bias_right)
    draw = posterior.as_density(method="gaussian").sample(density_key)
    density_left = Line(draw.slope_left, draw.bias_left)
    density_right = Line(draw.slope_right, draw.bias_right)

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(nrows=2, figsize=(11, 8))
    grid = jp.linspace(x.min(), x.max(), 300)
    kwargs = {"s": 30, "alpha": 0.8, "color": plot.BLUE_GREY.neutral}
    plot.scatter(x, observations, axes[0], **kwargs)
    plot_posterior_lines(axes[0], grid, (left, right), map_x, args.stdv)
    true_mean = compute_piecewise(grid, *true_lines, grid <= true_x)
    kwargs = {"linestyle": "--", "label": "true line"}
    plot.line(grid, true_mean, axes[0], plot.EARTH.secondary, **kwargs)
    density_lines = (density_left, density_right)
    density_mean = compute_piecewise(grid, *density_lines, grid <= map_x)
    kwargs = {"linestyle": "--", "label": "gaussian approx"}
    plot.line(grid, density_mean, axes[0], "purple", **kwargs)
    plot_switches(axes[0], map_x, true_x)
    axes[0].set_title("Change-point regression with a discrete switch")
    axes[0].legend(loc="upper left")
    plot.set_labels(axes[0], x="x", y="y")
    plot.clean(axes[0])

    width = switch_positions[1] - switch_positions[0]
    bar_args = (switch_positions, switch_posterior, axes[1])
    plot.bar(*bar_args, plot.DANDELION.primary, width=width)
    plot_switches(axes[1], map_x, true_x)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="upper left")
    plot.set_labels(axes[1], x="Switch location (x)", y="Posterior probability")
    plot.clean(axes[1])
    figure.tight_layout()
    plot.show()

    plot.trace_panel(samples._asdict(), title="MCMC traces of the lines")
    plot.show()

    for side, true_line in zip(("left", "right"), true_lines):
        names = (f"slope_{side}", f"bias_{side}")
        side_draws = {name: getattr(draws, name) for name in names}
        plot.corner(side_draws, dict(zip(names, true_line)))
        plot.show()

    figure, axis = plot.subplots()
    plot.diagnostics(acceptance_rates, axis, plot.DANDELION.primary)
    axis.set_title("Acceptance rates per chain")
    plot.show()
