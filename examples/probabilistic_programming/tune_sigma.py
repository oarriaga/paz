from collections import namedtuple

import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.plot as plot

tfd = tfp.distributions
tfb = tfp.bijectors
PosteriorResult = namedtuple(
    "PosteriorResult", ["posterior", "acceptance_rate"]
)
TrueParameters = namedtuple("TrueParameters", ["mean", "bias", "stdv"])


def build_model(inputs, low, high):
    def likelihood(inputs):
        def apply(mean, bias, stdv):
            return tfd.Normal(mean * inputs + bias, stdv)

        return apply

    mean = paz.Prior(tfd.Normal(0.0, 1.0), name="mean")
    bias = paz.Prior(tfd.Normal(0.0, 1.0), name="bias")
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = paz.Prior(
        tfd.Uniform(low, high), name="stdv", bijector=bijector
    )
    y_pred = paz.Observable(likelihood(inputs), name="y_pred")(
        mean, bias, stdv
    )
    return paz.PGM([mean, bias, stdv], [y_pred], "line"), bijector


def compute_posterior(model, key, data, num_samples, **infer_kwargs):
    posterior = model.infer(key, data, num_samples=num_samples, **infer_kwargs)
    acceptance_rate = jp.mean(posterior.infos.acceptance_rate)
    return PosteriorResult(posterior, acceptance_rate)


def setup_plot_style():
    plot.configure(fontsize=11)




def plot_posterior(
    inputs,
    observations,
    posterior,
    acceptance_rate,
    true_params,
    title,
    density_draw=None,
    num_draws=150,
):
    forward = posterior.samples
    mean_samples = forward.mean
    bias_samples = forward.bias
    stdv_samples = forward.stdv

    trace_steps = jp.arange(mean_samples.shape[0]) + 1
    trace_mean = mean_samples.mean(axis=1)
    trace_bias = bias_samples.mean(axis=1)
    trace_stdv = stdv_samples.mean(axis=1)

    flat_mean = mean_samples.reshape((-1,))
    flat_bias = bias_samples.reshape((-1,))

    predictions = flat_mean[:, None] * inputs[None, :] + flat_bias[:, None]
    percentiles = jp.percentile(predictions, jp.array([5.0, 95.0]), axis=0)
    mean_prediction = predictions.mean(axis=0)

    total_draws = flat_mean.shape[0]
    num_draws = int(min(num_draws, total_draws))
    draw_indices = jp.linspace(0, total_draws - 1, num_draws)
    draw_indices = draw_indices.astype(jp.int32)
    draw_means = flat_mean[draw_indices]
    draw_biases = flat_bias[draw_indices]
    predictive_draws = (
        draw_means[:, None] * inputs[None, :] + draw_biases[:, None]
    )

    trace_steps = paz.to_numpy(trace_steps)
    trace_mean = paz.to_numpy(trace_mean)
    trace_bias = paz.to_numpy(trace_bias)
    trace_stdv = paz.to_numpy(trace_stdv)
    flat_mean = paz.to_numpy(flat_mean)
    flat_bias = paz.to_numpy(flat_bias)
    inputs = paz.to_numpy(inputs)
    observations = paz.to_numpy(observations)
    predictive_draws = paz.to_numpy(predictive_draws)
    percentiles = paz.to_numpy(percentiles)
    mean_prediction = paz.to_numpy(mean_prediction)

    figure, axes = plot.subplots(nrows=2, ncols=2, figsize=(12, 8))
    figure.suptitle(f"{title} (acceptance {float(acceptance_rate):.3f})")

    plot.line(trace_steps, trace_mean, axes[0, 0], color="C0", label="mean")
    plot.line(trace_steps, trace_bias, axes[0, 0], color="C1", label="bias")
    plot.hline(true_params.mean, axes[0, 0], color="C0", linestyle="--",
               label="true mean")
    plot.hline(true_params.bias, axes[0, 0], color="C1", linestyle="--",
               label="true bias")
    plot.set_labels(axes[0, 0], x="Sample", y="Value")
    plot.legend(axes[0, 0], loc="upper right")
    plot.clean(axes[0, 0])
    axes[0, 0].set_title("Chain-averaged traces")

    plot.line(trace_steps, trace_stdv, axes[0, 1], color="C2")
    plot.hline(true_params.stdv, axes[0, 1], color="C2", linestyle="--",
               label="true stdv")
    plot.set_labels(axes[0, 1], x="Sample", y="Stdv")
    plot.legend(axes[0, 1], loc="upper right")
    plot.clean(axes[0, 1])
    axes[0, 1].set_title("Noise scale trace")

    plot.scatter(flat_mean, flat_bias, axes[1, 0], color="C0", alpha=0.2, s=10)
    axes[1, 0].scatter(true_params.mean, true_params.bias, color="black", s=80,
                       marker="*", label="true")
    plot.set_labels(axes[1, 0], x="mean", y="bias")
    plot.legend(axes[1, 0], loc="upper right")
    plot.clean(axes[1, 0])
    axes[1, 0].set_title("Posterior mean vs bias")

    for draw in predictive_draws:
        plot.line(inputs, draw, axes[1, 1], color="C0", alpha=0.05)
    plot.fill_between(inputs, percentiles[0], percentiles[1], axes[1, 1],
                      color="C0", alpha=0.2, label="90% band")
    plot.line(inputs, mean_prediction, axes[1, 1], color="C0", linewidth=2,
              label="posterior mean")
    plot.line(
        inputs,
        true_params.mean * inputs + true_params.bias,
        axes[1, 1],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="true mean line",
    )
    plot.scatter(inputs, observations, axes[1, 1], color="C3", s=16, alpha=0.8,
                 label="observations")
    if density_draw is not None:
        density_line = density_draw.mean * inputs + density_draw.bias
        plot.line(inputs, density_line, axes[1, 1], color="tab:purple",
                  linestyle="--", linewidth=2, label="gaussian approx")
    plot.set_labels(axes[1, 1], x="inputs", y="observations")
    plot.legend(axes[1, 1], loc="upper right")
    plot.clean(axes[1, 1])
    axes[1, 1].set_title("Posterior predictive")

    figure.tight_layout()
    plot.show()


inputs = jp.linspace(0.0, 1.0, 200)
observations = 0.5 * inputs + 0.1 + 0.05 * jp.sin(50.0 * inputs)
true_params = TrueParameters(mean=0.5, bias=0.1, stdv=0.05)
low, high = 0.001, 0.3
model, _ = build_model(inputs, low, high)
data = {"y_pred": observations}

num_chains = 4
num_tune_steps = 5000
num_episodes = 50
sigma_initial = 0.05
num_samples = 50_000
burn_in = 1000

setup_plot_style()

key = jax.random.PRNGKey(0)
key, tuned_key, initial_key, tuned_density_key, initial_density_key = (
    jax.random.split(key, 5)
)
model.configure(
    num_chains=num_chains,
    sigma=sigma_initial,
    warmup=burn_in,
    tuner=paz.AdaptiveStepTuner(
        sigma=sigma_initial,
        num_steps=num_tune_steps,
        num_episodes=num_episodes,
    ),
)

tuned_result = compute_posterior(model, tuned_key, data, num_samples)
tuned_sigma = tuned_result.posterior.config["sigma"]

initial_result = compute_posterior(
    model,
    initial_key,
    data,
    num_samples,
    sigma=sigma_initial,
    warmup=burn_in,
    num_chains=num_chains,
    tuner=None,
)

print("Tuning results")
print(f"  Tuned sigma (mean across chains): {tuned_sigma:.5f}")

tuned_forward = tuned_result.posterior.samples
initial_forward = initial_result.posterior.samples

tuned_density = tuned_result.posterior.as_density(method="gaussian")
tuned_density_draw = tuned_density.sample(
    tuned_density_key, num_samples=1
)
initial_density = initial_result.posterior.as_density(method="gaussian")
initial_density_draw = initial_density.sample(
    initial_density_key, num_samples=1
)

print("Posterior estimates (tuned sigma)")
print(
    "  Mean: "
    f"{tuned_result.posterior.inverse_samples.mean.mean():.4f} "
    f"(true: {true_params.mean:.2f})"
)
print(
    "  Bias: "
    f"{tuned_result.posterior.inverse_samples.bias.mean():.4f} "
    f"(true: {true_params.bias:.2f})"
)
print(
    "  Stdv: "
    f"{tuned_forward.stdv.mean():.4f} "
    f"(true: {true_params.stdv:.2f})"
)
print(f"  Acceptance rate: {tuned_result.acceptance_rate:.3f}")

print("Posterior estimates (initial sigma)")
print(
    "  Mean: "
    f"{initial_result.posterior.inverse_samples.mean.mean():.4f} "
    f"(true: {true_params.mean:.2f})"
)
print(
    "  Bias: "
    f"{initial_result.posterior.inverse_samples.bias.mean():.4f} "
    f"(true: {true_params.bias:.2f})"
)
print(
    "  Stdv: "
    f"{initial_forward.stdv.mean():.4f} "
    f"(true: {true_params.stdv:.2f})"
)
print(f"  Acceptance rate: {initial_result.acceptance_rate:.3f}")

plot_posterior(
    inputs,
    observations,
    tuned_result.posterior,
    tuned_result.acceptance_rate,
    true_params,
    "Posterior with tuned sigma",
    tuned_density_draw,
)
plot_posterior(
    inputs,
    observations,
    initial_result.posterior,
    initial_result.acceptance_rate,
    true_params,
    "Posterior with initial sigma",
    initial_density_draw,
)
