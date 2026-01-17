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

from collections import namedtuple

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.types import SampleType

tfd = tfp.distributions

LineParameters = namedtuple("LineParameters", ["slope", "bias"])
LineSet = namedtuple("LineSet", ["left", "right"])
SwitchPlotArgs = namedtuple(
    "SwitchPlotArgs",
    ["sigma", "switch_support", "switch_posterior", "switch_true_index"],
)


def build_switch_model(x, sigma):
    slope_left = paz.Prior(tfd.Normal(0.0, 1.0), name="slope_left")
    bias_left = paz.Prior(tfd.Normal(0.0, 1.0), name="bias_left")
    slope_right = paz.Prior(tfd.Normal(0.0, 1.0), name="slope_right")
    bias_right = paz.Prior(tfd.Normal(0.0, 1.0), name="bias_right")
    num_switch_positions = x.shape[0] - 1
    switch_index = paz.Prior(
        tfd.Categorical(
            logits=jp.zeros(num_switch_positions), dtype=jp.float32
        ),
        name="switch_index",
    )

    def y_distribution(
        slope_left, bias_left, slope_right, bias_right, switch_index
    ):
        indices = jp.arange(x.shape[0])
        switch_index = jp.asarray(switch_index, dtype=jp.int32)
        use_left = indices <= switch_index
        mean_left = slope_left * x + bias_left
        mean_right = slope_right * x + bias_right
        mean = jp.where(use_left, mean_left, mean_right)
        return tfd.Normal(mean, sigma)

    y_obs = paz.Observable(y_distribution, name="y")(
        slope_left, bias_left, slope_right, bias_right, switch_index
    )
    return paz.PGM(
        [slope_left, bias_left, slope_right, bias_right, switch_index],
        [y_obs],
        "change_point_switch",
    )


def run_mcmc(
    model_marg,
    key,
    data,
    num_samples,
    num_chains,
    step_sigma,
    warmup,
):
    tune_key, infer_key = jax.random.split(key)
    model_marg.tune(
        tune_key,
        data,
        num_chains=num_chains,
        sigma=step_sigma,
        warmup=warmup,
        num_samples=num_samples,
    )
    return model_marg.infer(infer_key, data, num_samples=num_samples)


def plot_switch_results(
    x,
    observations,
    posterior_lines,
    true_lines,
    plot_args,
    density_lines=None,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    x_grid = jp.linspace(x.min(), x.max(), 300)
    switch_positions = 0.5 * (x[:-1] + x[1:])
    switch_map_index = int(
        plot_args.switch_support[jp.argmax(plot_args.switch_posterior)]
    )
    switch_map_x = switch_positions[switch_map_index]
    switch_true_x = switch_positions[plot_args.switch_true_index]

    left_line = posterior_lines.left.slope * x_grid + posterior_lines.left.bias
    right_line = (
        posterior_lines.right.slope * x_grid + posterior_lines.right.bias
    )
    piecewise_line = jp.where(x_grid <= switch_map_x, left_line, right_line)
    upper = piecewise_line + 2.0 * plot_args.sigma
    lower = piecewise_line - 2.0 * plot_args.sigma

    true_left = true_lines.left.slope * x_grid + true_lines.left.bias
    true_right = true_lines.right.slope * x_grid + true_lines.right.bias
    true_piecewise = jp.where(x_grid <= switch_true_x, true_left, true_right)

    axes[0].scatter(x, observations, color="tab:gray", alpha=0.8, label="data")
    axes[0].plot(x_grid, piecewise_line, color="black", label="posterior mean")
    axes[0].fill_between(
        x_grid,
        lower,
        upper,
        color="tab:blue",
        alpha=0.15,
        label=f"noise band (sigma={plot_args.sigma})",
    )
    axes[0].plot(
        x_grid,
        left_line,
        color="tab:blue",
        linestyle=":",
        label="left line (posterior)",
    )
    axes[0].plot(
        x_grid,
        right_line,
        color="tab:orange",
        linestyle=":",
        label="right line (posterior)",
    )
    axes[0].plot(
        x_grid,
        true_piecewise,
        color="tab:green",
        linestyle="--",
        label="true line",
    )
    if density_lines is not None:
        density_left = (
            density_lines.left.slope * x_grid + density_lines.left.bias
        )
        density_right = (
            density_lines.right.slope * x_grid + density_lines.right.bias
        )
        density_piecewise = jp.where(
            x_grid <= switch_map_x, density_left, density_right
        )
        axes[0].plot(
            x_grid,
            density_piecewise,
            color="tab:purple",
            linestyle="--",
            label="gaussian approx",
        )
    axes[0].axvline(
        switch_map_x,
        color="black",
        linewidth=2,
        label="posterior switch",
    )
    axes[0].axvline(
        switch_true_x,
        color="tab:green",
        linestyle="--",
        label="true switch",
    )
    axes[0].set_title("Change-point regression with a discrete switch")
    axes[0].legend(loc="upper left")

    axes[1].plot(
        switch_positions,
        plot_args.switch_posterior,
        color="tab:purple",
        marker="o",
    )
    axes[1].axvline(
        switch_map_x,
        color="black",
        linewidth=2,
        label="posterior switch",
    )
    axes[1].axvline(
        switch_true_x,
        color="tab:green",
        linestyle="--",
        label="true switch",
    )
    axes[1].set_xlabel("Switch location (x)")
    axes[1].set_ylabel("Posterior probability")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    return fig


def main():
    key = jax.random.PRNGKey(21)
    num_observations = 40
    x = jp.linspace(-1.0, 1.0, num_observations)
    sigma = 0.15

    slope_left_true = 1.0
    bias_left_true = 0.2
    slope_right_true = -0.6
    bias_right_true = 0.8
    switch_index_true = 17

    indices = jp.arange(num_observations)
    use_left = indices <= switch_index_true
    mean_left = slope_left_true * x + bias_left_true
    mean_right = slope_right_true * x + bias_right_true
    mean = jp.where(use_left, mean_left, mean_right)
    key, noise_key = jax.random.split(key)
    observations = mean + sigma * jax.random.normal(
        noise_key, (num_observations,)
    )

    model = build_switch_model(x, sigma)
    model_marg = paz.marginalize(model, ["switch_index"])
    data = {"y": observations}

    num_samples = 10_000
    num_chains = 10
    burn_in = 2000
    step_sigma = 0.25

    posterior = run_mcmc(
        model_marg, key, data, num_samples, num_chains, step_sigma, burn_in
    )
    samples, infos = posterior.samples, posterior.infos
    slope_left_samples = samples.position.slope_left.reshape(-1)
    bias_left_samples = samples.position.bias_left.reshape(-1)
    slope_right_samples = samples.position.slope_right.reshape(-1)
    bias_right_samples = samples.position.bias_right.reshape(-1)

    slope_left_mean = slope_left_samples.mean()
    bias_left_mean = bias_left_samples.mean()
    slope_right_mean = slope_right_samples.mean()
    bias_right_mean = bias_right_samples.mean()
    acceptance_rate = infos.acceptance_rate.mean()

    Theta = SampleType(["slope_left", "bias_left", "slope_right", "bias_right"])
    theta_samples = Theta(
        slope_left_samples,
        bias_left_samples,
        slope_right_samples,
        bias_right_samples,
    )
    switch_posterior = paz.recover_discrete_posterior(
        model_marg, "switch_index", theta_samples, data
    )
    switch_support = switch_posterior.support
    switch_probs = switch_posterior.posterior.mean(axis=0)

    posterior_lines = LineSet(
        LineParameters(slope_left_mean, bias_left_mean),
        LineParameters(slope_right_mean, bias_right_mean),
    )
    density = posterior.as_density(method="gaussian")
    key, density_key = jax.random.split(key)
    density_sample = density.sample(density_key, num_samples=1)
    density_lines = LineSet(
        LineParameters(density_sample.slope_left, density_sample.bias_left),
        LineParameters(density_sample.slope_right, density_sample.bias_right),
    )
    true_lines = LineSet(
        LineParameters(slope_left_true, bias_left_true),
        LineParameters(slope_right_true, bias_right_true),
    )
    plot_args = SwitchPlotArgs(
        sigma, switch_support, switch_probs, switch_index_true
    )

    switch_map_index = int(switch_support[jp.argmax(switch_probs)])
    switch_positions = 0.5 * (x[:-1] + x[1:])
    switch_map_x = switch_positions[switch_map_index]
    switch_true_x = switch_positions[switch_index_true]

    print("=" * 60)
    print("Change-point regression with discrete switch")
    print("=" * 60)
    print(f"true switch index={switch_index_true}, x={switch_true_x:.3f}")
    print(f"posterior switch index={switch_map_index}, x={switch_map_x:.3f}")
    print(f"acceptance rate={acceptance_rate:.3f}")

    plot_switch_results(
        x,
        observations,
        posterior_lines,
        true_lines,
        plot_args,
        density_lines,
    )
    plt.show()


if __name__ == "__main__":
    main()
