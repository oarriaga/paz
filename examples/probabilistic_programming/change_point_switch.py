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
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.plot as plot
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
    model_marg.configure(
        num_chains=num_chains,
        warmup=warmup,
        sigma=step_sigma,
        tuner=paz.AdaptiveStepTuner(sigma=step_sigma),
    )
    return model_marg.infer(key, data, num_samples=num_samples)


def plot_switch_results(
    x,
    observations,
    posterior_lines,
    true_lines,
    plot_args,
    density_lines=None,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    x_np = np.array(x)
    obs_np = np.array(observations)
    x_grid = np.linspace(float(x.min()), float(x.max()), 300)
    switch_positions = 0.5 * (x_np[:-1] + x_np[1:])
    switch_map_index = int(
        plot_args.switch_support[jp.argmax(plot_args.switch_posterior)]
    )
    switch_map_x = switch_positions[switch_map_index]
    switch_true_x = switch_positions[plot_args.switch_true_index]

    left_line = (
        float(posterior_lines.left.slope) * x_grid
        + float(posterior_lines.left.bias)
    )
    right_line = (
        float(posterior_lines.right.slope) * x_grid
        + float(posterior_lines.right.bias)
    )
    piecewise_line = np.where(x_grid <= switch_map_x, left_line, right_line)
    upper = piecewise_line + 2.0 * plot_args.sigma
    lower = piecewise_line - 2.0 * plot_args.sigma

    true_left = true_lines.left.slope * x_grid + true_lines.left.bias
    true_right = true_lines.right.slope * x_grid + true_lines.right.bias
    true_piecewise = np.where(x_grid <= switch_true_x, true_left, true_right)

    # Regression plot
    plot.scatter(
        x_np, obs_np, axes[0], s=30, alpha=0.8, color=plot.BLUE_GREY.neutral
    )
    plot.line(
        x_grid, piecewise_line, axes[0], color="black", label="posterior mean"
    )
    plot.line_with_band(
        x_grid,
        piecewise_line,
        np.full_like(piecewise_line, plot_args.sigma),
        axes[0],
        color=plot.BLUE_GREY.primary,
        alpha=0.15,
        label=f"noise band (sigma={plot_args.sigma})",
    )
    plot.line(x_grid, left_line, axes[0], color=plot.BLUE_GREY.primary,
              linestyle=":", label="left line (posterior)")
    plot.line(x_grid, right_line, axes[0], color=plot.EARTH.accent,
              linestyle=":", label="right line (posterior)")
    plot.line(x_grid, true_piecewise, axes[0], color=plot.EARTH.secondary,
              linestyle="--", label="true line")

    if density_lines is not None:
        density_left = (
            float(density_lines.left.slope) * x_grid
            + float(density_lines.left.bias)
        )
        density_right = (
            float(density_lines.right.slope) * x_grid
            + float(density_lines.right.bias)
        )
        density_piecewise = np.where(
            x_grid <= switch_map_x, density_left, density_right
        )
        plot.line(x_grid, density_piecewise, axes[0], color="purple",
                  linestyle="--", label="gaussian approx")

    plot.vline(switch_map_x, axes[0], color="black", linewidth=2,
               label="posterior switch")
    plot.vline(switch_true_x, axes[0], color=plot.EARTH.secondary,
               linestyle="--", label="true switch")
    axes[0].set_title("Change-point regression with a discrete switch")
    axes[0].legend(loc="upper left")
    plot.set_labels(axes[0], x="x", y="y")
    plot.clean(axes[0])

    # Discrete posterior plot
    switch_probs_np = np.array(plot_args.switch_posterior)
    plot.discrete_posterior(switch_positions, switch_probs_np, axes[1],
                            true_value=switch_true_x,
                            color=plot.DANDELION.primary,
                            true_color=plot.EARTH.secondary)
    plot.vline(switch_map_x, axes[1], color="black", linewidth=2,
               label="MAP switch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="upper left")
    plot.set_labels(axes[1], x="Switch location (x)", y="Posterior probability")
    plot.clean(axes[1])

    plt.tight_layout()
    return fig


# Configure plotting
plot.configure(fontsize=12, latex=False)

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
samples, infos = posterior.inverse_samples, posterior.infos
slope_left_samples = samples.slope_left.reshape(-1)
bias_left_samples = samples.bias_left.reshape(-1)
slope_right_samples = samples.slope_right.reshape(-1)
bias_right_samples = samples.bias_right.reshape(-1)

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
switch_positions = 0.5 * (np.array(x)[:-1] + np.array(x)[1:])
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
plot.show()

# Additional plots: Trace panel for line parameters
plot.trace_panel({
    "slope_left": samples.slope_left,
    "bias_left": samples.bias_left,
    "slope_right": samples.slope_right,
    "bias_right": samples.bias_right,
}, title="MCMC Traces for Line Parameters")
plot.show()

# Corner plot for left line parameters
plot.corner(
    {"slope_left": np.array(slope_left_samples),
     "bias_left": np.array(bias_left_samples)},
    true_values={"slope_left": slope_left_true, "bias_left": bias_left_true}
)
plot.show()

# Corner plot for right line parameters
plot.corner(
    {"slope_right": np.array(slope_right_samples),
     "bias_right": np.array(bias_right_samples)},
    true_values={"slope_right": slope_right_true, "bias_right": bias_right_true}
)
plot.show()

# Diagnostics
fig, ax = plot.subplots()
plot.diagnostics(infos.acceptance_rate, ax, color=plot.DANDELION.primary)
ax.set_title("Acceptance rates per chain")
plot.clean(ax)
plot.show()
