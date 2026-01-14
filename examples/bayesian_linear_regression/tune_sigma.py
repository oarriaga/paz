from collections import namedtuple

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz

tfd = tfp.distributions
tfb = tfp.bijectors
PosteriorResult = namedtuple(
    "PosteriorResult", ["posterior", "acceptance_rate"]
)
TrueParameters = namedtuple("TrueParameters", ["mean", "bias", "stdv"])


def build_model(inputs, observations, low, high):
    def likelihood(inputs):
        def apply(mean, bias, stdv):
            return tfd.Normal(mean * inputs + bias, stdv)

        return apply

    mean = paz.Prior("mean", tfd.Normal(0.0, 1.0))
    bias = paz.Prior("bias", tfd.Normal(0.0, 1.0))
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    stdv = paz.Prior("stdv", tfd.Uniform(low, high), bijector=bijector)
    y_pred = paz.Observable("y_pred", likelihood(inputs), observations)(
        mean, bias, stdv
    )
    return paz.PGM([mean, bias, stdv], [y_pred], "line"), bijector


def compute_posterior(
    key, log_density_fn, positions, sigma, num_samples, num_chains, burn_in
):
    samples, infos = paz.metropolis_hastings.sample(
        key,
        log_density_fn,
        positions,
        sigma,
        num_samples,
        num_chains,
    )
    acceptance_rate = jp.mean(infos.acceptance_rate[burn_in:])
    posterior = jax.tree.map(lambda x: x[burn_in:], samples)
    return PosteriorResult(posterior, acceptance_rate)


def setup_plot_style():
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
        }
    )


def plot_tuning(tune_infos, tuned_sigma, sigma_initial):
    acceptance_rate = tune_infos.acceptance_rate
    sigma_history = tune_infos.sigma * tune_infos.factor
    episode_ids = jp.arange(1, acceptance_rate.shape[0] + 1)
    chain_count = acceptance_rate.shape[1]

    acceptance_mean = acceptance_rate.mean(axis=1)
    sigma_mean = sigma_history.mean(axis=1)

    episode_ids = paz.to_numpy(episode_ids)
    acceptance_rate = paz.to_numpy(acceptance_rate)
    sigma_history = paz.to_numpy(sigma_history)
    acceptance_mean = paz.to_numpy(acceptance_mean)
    sigma_mean = paz.to_numpy(sigma_mean)

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for chain_index in range(chain_count):
        axes[0].plot(
            episode_ids,
            acceptance_rate[:, chain_index],
            color="C0",
            alpha=0.2,
        )
    axes[0].plot(
        episode_ids,
        acceptance_mean,
        color="C0",
        linewidth=2,
        label="mean acceptance",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Acceptance rate")
    axes[0].set_title("Tuning acceptance by episode")
    axes[0].legend(loc="lower right")

    for chain_index in range(chain_count):
        axes[1].plot(
            episode_ids,
            sigma_history[:, chain_index],
            color="C1",
            alpha=0.2,
        )
    axes[1].plot(
        episode_ids,
        sigma_mean,
        color="C1",
        linewidth=2,
        label="mean sigma",
    )
    axes[1].axhline(
        sigma_initial,
        color="0.4",
        linestyle="--",
        linewidth=1.5,
        label="initial sigma",
    )
    axes[1].axhline(
        float(tuned_sigma),
        color="C1",
        linestyle=":",
        linewidth=1.5,
        label="tuned sigma",
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Proposal sigma")
    axes[1].set_title("Sigma adaptation during tuning")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_posterior(
    inputs,
    observations,
    posterior,
    acceptance_rate,
    bijector,
    true_params,
    title,
    num_draws=150,
):
    mean_samples = posterior.position.mean
    bias_samples = posterior.position.bias
    stdv_samples = bijector(posterior.position.stdv)

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

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(f"{title} (acceptance {float(acceptance_rate):.3f})")

    axes[0, 0].plot(trace_steps, trace_mean, color="C0", label="mean")
    axes[0, 0].plot(trace_steps, trace_bias, color="C1", label="bias")
    axes[0, 0].axhline(
        true_params.mean,
        color="C0",
        linestyle="--",
        linewidth=1.2,
        label="true mean",
    )
    axes[0, 0].axhline(
        true_params.bias,
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label="true bias",
    )
    axes[0, 0].set_title("Chain-averaged traces")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(trace_steps, trace_stdv, color="C2")
    axes[0, 1].axhline(
        true_params.stdv,
        color="C2",
        linestyle="--",
        linewidth=1.2,
        label="true stdv",
    )
    axes[0, 1].set_title("Noise scale trace")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Stdv")
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].scatter(
        flat_mean,
        flat_bias,
        color="C0",
        alpha=0.2,
        s=10,
    )
    axes[1, 0].scatter(
        true_params.mean,
        true_params.bias,
        color="black",
        s=80,
        marker="*",
        label="true",
    )
    axes[1, 0].set_title("Posterior mean vs bias")
    axes[1, 0].set_xlabel("mean")
    axes[1, 0].set_ylabel("bias")
    axes[1, 0].legend(loc="upper right")

    for draw in predictive_draws:
        axes[1, 1].plot(inputs, draw, color="C0", alpha=0.05)
    axes[1, 1].fill_between(
        inputs,
        percentiles[0],
        percentiles[1],
        color="C0",
        alpha=0.2,
        label="90% band",
    )
    axes[1, 1].plot(
        inputs,
        mean_prediction,
        color="C0",
        linewidth=2,
        label="posterior mean",
    )
    axes[1, 1].plot(
        inputs,
        true_params.mean * inputs + true_params.bias,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="true mean line",
    )
    axes[1, 1].scatter(
        inputs,
        observations,
        color="C3",
        s=16,
        alpha=0.8,
        label="observations",
    )
    axes[1, 1].set_title("Posterior predictive")
    axes[1, 1].set_xlabel("inputs")
    axes[1, 1].set_ylabel("observations")
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


inputs = jp.linspace(0.0, 1.0, 200)
observations = 0.5 * inputs + 0.1 + 0.05 * jp.sin(50.0 * inputs)
true_params = TrueParameters(mean=0.5, bias=0.1, stdv=0.05)
low, high = 0.001, 0.3
model, bijector = build_model(inputs, observations, low, high)

log_density_fn = lambda params: model.apply(params).log_prob_sum

num_chains = 4
num_tune_steps = 5000
num_episodes = 50
sigma_initial = 0.05
num_samples = 50_000
burn_in = 1000

setup_plot_style()

key = jax.random.PRNGKey(0)
key, init_key, tune_key, tuned_key, initial_key = jax.random.split(key, 5)
positions = model.sample_inverse(init_key, num_chains)

tune = paz.Tuner(log_density_fn, positions, num_chains)
tuned_sigma, tune_infos = tune(
    tune_key, num_tune_steps, num_episodes, sigma_initial
)

episode_acceptance = tune_infos.acceptance_rate.mean(axis=1)
print("Tuning results")
print(f"  Final episode acceptance rate: {episode_acceptance[-1]:.3f}")
print(f"  Tuned sigma (mean across chains): {tuned_sigma:.5f}")

plot_tuning(tune_infos, tuned_sigma, sigma_initial)

tuned_result = compute_posterior(
    tuned_key,
    log_density_fn,
    positions,
    tuned_sigma,
    num_samples,
    num_chains,
    burn_in,
)
initial_result = compute_posterior(
    initial_key,
    log_density_fn,
    positions,
    sigma_initial,
    num_samples,
    num_chains,
    burn_in,
)

print("Posterior estimates (tuned sigma)")
print(
    "  Mean: "
    f"{tuned_result.posterior.position.mean.mean():.4f} "
    f"(true: {true_params.mean:.2f})"
)
print(
    "  Bias: "
    f"{tuned_result.posterior.position.bias.mean():.4f} "
    f"(true: {true_params.bias:.2f})"
)
print(
    "  Stdv: "
    f"{bijector(tuned_result.posterior.position.stdv).mean():.4f} "
    f"(true: {true_params.stdv:.2f})"
)
print(f"  Acceptance rate: {tuned_result.acceptance_rate:.3f}")

print("Posterior estimates (initial sigma)")
print(
    "  Mean: "
    f"{initial_result.posterior.position.mean.mean():.4f} "
    f"(true: {true_params.mean:.2f})"
)
print(
    "  Bias: "
    f"{initial_result.posterior.position.bias.mean():.4f} "
    f"(true: {true_params.bias:.2f})"
)
print(
    "  Stdv: "
    f"{bijector(initial_result.posterior.position.stdv).mean():.4f} "
    f"(true: {true_params.stdv:.2f})"
)
print(f"  Acceptance rate: {initial_result.acceptance_rate:.3f}")

plot_posterior(
    inputs,
    observations,
    tuned_result.posterior,
    tuned_result.acceptance_rate,
    bijector,
    true_params,
    "Posterior with tuned sigma",
)
plot_posterior(
    inputs,
    observations,
    initial_result.posterior,
    initial_result.acceptance_rate,
    bijector,
    true_params,
    "Posterior with initial sigma",
)
