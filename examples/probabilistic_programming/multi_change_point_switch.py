# Multi change-point regression with a discrete switch configuration.
#
# This generalizes the single switch example by using a single discrete
# variable that indexes a table of all ordered switch configurations.
# The support size grows combinatorially, so keep the dataset small.

import argparse
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jp

import paz
import paz.utils.plot as plot


def build_switch_table(num_observations, num_switches):
    positions = range(num_observations - 1)
    table = list(combinations(positions, num_switches))
    return jp.array(table, dtype=jp.int32)


def build_switch_model(x, stdv, switch_table):
    num_segments = switch_table.shape[1] + 1
    logits = jp.zeros(len(switch_table))
    switch = paz.distributions.Categorical(logits=logits, dtype=jp.float32)
    priors = [paz.Prior(switch, name="switch_index")]
    standard = paz.distributions.Normal(0.0, 1.0)
    for prefix in ("slope", "bias"):
        for segment in range(num_segments):
            name = f"{prefix}_segment_{segment}"
            priors.append(paz.Prior(standard, name=name))
    likelihood = Likelihood(x, stdv, switch_table)
    y = paz.Observable(likelihood, name="y")(*priors)
    return paz.PGM(priors, [y], "multi_change_point_switch")


def Likelihood(x, stdv, switch_table):
    def apply(switch_index, *lines):
        slopes, biases = jp.split(jp.stack(lines), 2)
        switch_indices = switch_table[switch_index.astype(jp.int32)]
        mean = compute_piecewise_mean(x, slopes, biases, switch_indices)
        return paz.distributions.Normal(mean, stdv)

    return apply


def unpack_lines(sample, num_segments):
    lines = []
    for prefix in ("slope", "bias"):
        names = [f"{prefix}_segment_{arg}" for arg in range(num_segments)]
        lines.append(jp.stack([getattr(sample, name) for name in names]))
    return lines


def compute_piecewise_mean(x, slopes, biases, switch_indices):
    segments = compute_segments(len(x), switch_indices)
    return slopes[segments] * x + biases[segments]


def compute_segments(num_observations, switch_indices):
    indices = jp.arange(num_observations)
    return (switch_indices[:, None] < indices).sum(axis=0)


def plot_fit(axis, x, observations, posterior_mean, density_mean, true_mean):
    kwargs = {"color": "tab:gray", "alpha": 0.8, "label": "data"}
    plot.scatter(x, observations, axis, **kwargs)
    plot.line(x, posterior_mean, axis, "black", label="posterior mean")
    kwargs = {"linestyle": "--", "label": "gaussian approx"}
    plot.line(x, density_mean, axis, "tab:purple", **kwargs)
    kwargs = {"linestyle": "--", "label": "true mean"}
    plot.line(x, true_mean, axis, "tab:green", **kwargs)
    plot.legend(axis, loc="upper left")
    plot.clean(axis)
    axis.set_title("Piecewise regression with multiple switches")


def plot_switches(axis, map_positions, true_positions):
    for position in map_positions:
        plot.vline(position, axis, color="black")
    for position in true_positions:
        plot.vline(position, axis, color="tab:green", linewidth=1.2)


def plot_position_posterior(axis, posterior, switch_positions):
    num_switches = len(posterior)
    x_range = (switch_positions[0], switch_positions[-1])
    extent = [*x_range, 0.5, num_switches + 0.5]
    kwargs = {"extent": extent, "colorbar_label": "Posterior probability"}
    plot.imshow(posterior, axis, **kwargs)
    plot.set_labels(axis, x="Switch location (x)", y="Switch index")
    axis.set_yticks(range(1, num_switches + 1))
    axis.set_title("Posterior over switch locations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi change-point model")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num_switches", type=int, default=2)
    parser.add_argument("--num_observations", type=int, default=30)
    parser.add_argument("--stdv", type=float, default=0.15)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_chains", type=int, default=5)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.25)
    args = parser.parse_args()

    num_segments = args.num_switches + 1
    true_slopes = jp.linspace(1.1, -0.6, num_segments)
    true_biases = jp.linspace(0.8, -0.8, num_segments)
    # Evenly spaced switches give every segment enough points to locate it.
    last_index = args.num_observations - 1
    true_switches = jp.linspace(0, last_index, num_segments + 1)[1:-1]
    true_switches = jp.round(true_switches).astype(jp.int32)
    x = jp.linspace(-1.0, 1.0, args.num_observations)
    true_lines = (true_slopes, true_biases)
    true_mean = compute_piecewise_mean(x, *true_lines, true_switches)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    noise_key, infer_key, density_key = keys
    noise = args.stdv * jax.random.normal(noise_key, x.shape)
    observations = true_mean + noise

    switch_table = build_switch_table(args.num_observations, args.num_switches)
    switch_model = build_switch_model(x, args.stdv, switch_table)
    model = paz.marginalize(switch_model, ["switch_index"])
    tuner = paz.AdaptiveStepTuner(args.sigma)
    kwargs = {"warmup": args.warmup, "sigma": args.sigma, "tuner": tuner}
    model.configure(num_chains=args.num_chains, **kwargs)
    kwargs = {"num_samples": args.num_samples}
    posterior = model.infer(infer_key, observations, **kwargs)
    draws = jax.tree.map(jp.ravel, posterior.inverse_samples)
    switch_args = (switch_model, "switch_index", draws, observations)
    switch = paz.recover_discrete_posterior(*switch_args)
    config_posterior = switch.posterior.mean(axis=0)
    map_switches = switch_table[jp.argmax(config_posterior)]

    num_positions = args.num_observations - 1
    kwargs = {"weights": config_posterior, "length": num_positions}
    count_positions = partial(jp.bincount, **kwargs)
    position_posterior = jax.vmap(count_positions)(switch_table.T)
    mean_lines = unpack_lines(jax.tree.map(jp.mean, draws), num_segments)
    posterior_mean = compute_piecewise_mean(x, *mean_lines, map_switches)
    draw = posterior.as_density(method="gaussian").sample(density_key)
    density_lines = unpack_lines(draw, num_segments)
    density_mean = compute_piecewise_mean(x, *density_lines, map_switches)

    print(f"num_configs={len(switch_table)}")
    print(f"true switch indices={true_switches.tolist()}")
    print(f"map switch indices={map_switches.tolist()}")
    print(f"acceptance rate={posterior.infos.acceptance_rate.mean():.3f}")

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(nrows=2, figsize=(11, 8))
    means = (posterior_mean, density_mean, true_mean)
    plot_fit(axes[0], x, observations, *means)
    switch_positions = 0.5 * (x[:-1] + x[1:])
    map_positions = switch_positions[map_switches]
    true_positions = switch_positions[true_switches]
    plot_switches(axes[0], map_positions, true_positions)
    plot_position_posterior(axes[1], position_posterior, switch_positions)
    figure.tight_layout()
    plot.show()
