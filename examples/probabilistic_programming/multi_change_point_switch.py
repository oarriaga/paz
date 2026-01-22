# Multi change-point regression with a discrete switch configuration.
#
# This generalizes the single switch example by using a single discrete
# variable that indexes a table of all ordered switch configurations.
# The support size grows combinatorially, so keep the dataset small.

from collections import namedtuple
from itertools import combinations
import time

import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.plot as plot
from paz.inference.types import SampleType

tfd = tfp.distributions

RunResult = namedtuple(
    "RunResult",
    [
        "slope_means",
        "bias_means",
        "posterior_configs",
        "switch_table",
        "density_sample",
        "acceptance_rate",
        "mcmc_seconds",
        "posterior_seconds",
    ],
)


def run_inference(
    model_marg,
    key,
    num_switches,
    switch_table,
    data,
    num_samples,
    num_chains,
    burn_in,
    step_sigma,
):
    print("Running MCMC...")
    start_time = time.perf_counter()
    posterior = run_mcmc(
        model_marg, key, data, num_samples, num_chains, step_sigma, burn_in
    )
    samples, infos = posterior.inverse_samples, posterior.infos
    mcmc_seconds = time.perf_counter() - start_time
    print(f"mcmc: {mcmc_seconds:.3f}s")

    print("Extracting segment samples...")
    start_time = time.perf_counter()
    slope_samples, bias_samples = extract_segment_samples(
        samples.position, num_switches
    )
    slope_means = jp.array([samples.mean() for samples in slope_samples])
    bias_means = jp.array([samples.mean() for samples in bias_samples])
    print(f"segment extraction: {time.perf_counter() - start_time:.3f}s")

    print("Building theta samples...")
    start_time = time.perf_counter()
    theta_samples = build_theta_samples(
        slope_samples, bias_samples, num_switches
    )
    print(f"theta samples build: {time.perf_counter() - start_time:.3f}s")

    print("Recovering switch posterior...")
    start_time = time.perf_counter()
    posterior_configs = paz.recover_discrete_posterior(
        model_marg, "switch_index", theta_samples, data
    ).posterior.mean(axis=0)
    posterior_seconds = time.perf_counter() - start_time
    print(f"switch posterior: {posterior_seconds:.3f}s")

    acceptance_rate = infos.acceptance_rate.mean()
    density = posterior.as_density(method="gaussian")
    key, density_key = jax.random.split(key)
    density_sample = density.sample(density_key, num_samples=1)

    return RunResult(
        slope_means,
        bias_means,
        posterior_configs,
        switch_table,
        density_sample,
        acceptance_rate,
        mcmc_seconds,
        posterior_seconds,
    )


def plot_results(
    x,
    observations,
    posterior_mean,
    density_mean,
    true_mean,
    switch_positions,
    map_switch_indices,
    true_switch_indices,
    switch_position_posterior,
):
    num_switches = switch_position_posterior.shape[0]
    plot.configure(fontsize=12)
    figure, axes = plot.subplots(nrows=2, ncols=1, figsize=(11, 8))

    plot.scatter(
        x, observations, axes[0], color="tab:gray", alpha=0.8, label="data"
    )
    plot.line(x, posterior_mean, axes[0], color="black", label="posterior mean")
    plot.line(x, density_mean, axes[0], color="tab:purple", linestyle="--",
              label="gaussian approx")
    plot.line(x, true_mean, axes[0], color="tab:green", linestyle="--",
              label="true mean")
    for position in switch_positions[map_switch_indices]:
        plot.vline(position, axes[0], color="black", linewidth=1.5)
    for position in switch_positions[true_switch_indices]:
        plot.vline(position, axes[0], color="tab:green", linestyle="--",
                   linewidth=1.2)
    plot.legend(axes[0], loc="upper left")
    plot.clean(axes[0])
    axes[0].set_title("Piecewise regression with multiple switches")

    extent = [
        float(switch_positions[0]),
        float(switch_positions[-1]),
        0.5,
        num_switches + 0.5,
    ]
    plot.imshow(switch_position_posterior, axes[1], cmap="viridis",
                aspect="auto", origin="lower", extent=extent,
                colorbar=True, colorbar_label="Posterior probability")
    plot.set_labels(axes[1], x="Switch location (x)", y="Switch index")
    axes[1].set_yticks(range(1, num_switches + 1))
    axes[1].set_title("Posterior over switch locations")
    figure.tight_layout()


def run_mcmc(
    model_marg, key, data, num_samples, num_chains, step_sigma, warmup
):
    model_marg.configure(
        num_chains=num_chains,
        warmup=warmup,
        sigma=step_sigma,
        tuner=paz.AdaptiveStepTuner(sigma=step_sigma),
    )
    return model_marg.infer(key, data, num_samples=num_samples)


def build_switch_model(x, sigma, switch_table, num_switches):
    num_segments = num_switches + 1
    slope_priors = []
    bias_priors = []
    for segment_index in range(num_segments):
        slope_priors.append(
            paz.Prior(
                tfd.Normal(0.0, 1.0),
                name=f"slope_segment_{segment_index}",
            )
        )
        bias_priors.append(
            paz.Prior(
                tfd.Normal(0.0, 1.0),
                name=f"bias_segment_{segment_index}",
            )
        )

    switch_index = paz.Prior(
        tfd.Categorical(
            logits=jp.zeros(switch_table.shape[0]), dtype=jp.float32
        ),
        name="switch_index",
    )

    def y_distribution(switch_index, *segment_params):
        slopes = jp.stack(segment_params[:num_segments])
        biases = jp.stack(segment_params[num_segments:])
        index = jp.asarray(switch_index, dtype=jp.int32)
        switch_indices = switch_table[index]
        mean = compute_piecewise_mean(x, slopes, biases, switch_indices)
        return tfd.Normal(mean, sigma)

    y_obs = paz.Observable(y_distribution, name="y")(
        switch_index, *slope_priors, *bias_priors
    )

    inputs = [switch_index] + slope_priors + bias_priors
    return paz.PGM(inputs, [y_obs], "multi_change_point_switch")


def build_switch_table(num_observations, num_switches):
    positions = range(num_observations - 1)
    table = list(combinations(positions, num_switches))
    return jp.array(table, dtype=jp.int32)


def build_true_segments(num_switches):
    num_segments = num_switches + 1
    slopes = jp.linspace(1.1, -0.6, num_segments)
    biases = jp.linspace(0.4, -0.2, num_segments)
    return slopes, biases


def build_true_switch_indices(num_observations, num_switches):
    max_index = num_observations - 2
    positions = jp.linspace(1, max_index, num_switches)
    positions = jp.round(positions).astype(jp.int32)
    positions = jp.unique(positions)
    if positions.shape[0] != num_switches:
        positions = jp.arange(1, num_switches + 1, dtype=jp.int32)
    return positions


def compute_piecewise_mean(x, slopes, biases, switch_indices):
    num_observations = x.shape[0]
    segment_ids = compute_segment_ids(num_observations, switch_indices)
    return slopes[segment_ids] * x + biases[segment_ids]


def compute_segment_ids(num_observations, switch_indices):
    indices = jp.arange(num_observations)
    switch_indices = jp.asarray(switch_indices, dtype=jp.int32)
    switch_indices = jp.sort(switch_indices)
    return (switch_indices[:, None] < indices[None, :]).sum(axis=0)


def extract_segment_samples(position, num_switches):
    num_segments = num_switches + 1
    slope_samples = []
    bias_samples = []
    for segment_index in range(num_segments):
        slope_name = f"slope_segment_{segment_index}"
        bias_name = f"bias_segment_{segment_index}"
        slope_samples.append(getattr(position, slope_name).reshape(-1))
        bias_samples.append(getattr(position, bias_name).reshape(-1))
    return slope_samples, bias_samples


def extract_segment_values(sample, num_switches):
    num_segments = num_switches + 1
    slopes = []
    biases = []
    for segment_index in range(num_segments):
        slope_name = f"slope_segment_{segment_index}"
        bias_name = f"bias_segment_{segment_index}"
        slopes.append(getattr(sample, slope_name))
        biases.append(getattr(sample, bias_name))
    return jp.stack(slopes), jp.stack(biases)


def build_theta_samples(slope_samples, bias_samples, num_switches):
    num_segments = num_switches + 1
    names = []
    values = []
    for segment_index in range(num_segments):
        names.append(f"slope_segment_{segment_index}")
        values.append(slope_samples[segment_index])
    for segment_index in range(num_segments):
        names.append(f"bias_segment_{segment_index}")
        values.append(bias_samples[segment_index])
    Theta = SampleType(names)
    return Theta(*values)


def compute_switch_position_posterior(
    switch_table, posterior_configs, num_positions
):
    num_switches = switch_table.shape[1]
    posterior = jp.zeros((num_switches, num_positions))
    for switch_index in range(num_switches):
        positions = switch_table[:, switch_index]
        posterior = posterior.at[switch_index, positions].add(posterior_configs)
    return posterior


key = jax.random.PRNGKey(777)
num_switches = 2
num_observations = 30
sigma = 0.15
num_samples = 5_000
num_chains = 5
burn_in = 500
step_sigma = 0.25

x = jp.linspace(-1.0, 1.0, num_observations)
print("Building ground-truth segments...")
start_time = time.perf_counter()
true_slopes, true_biases = build_true_segments(num_switches)
true_switch_indices = build_true_switch_indices(
    num_observations, num_switches
)
print(f"ground-truth build: {time.perf_counter() - start_time:.3f}s")

key, noise_key = jax.random.split(key)
print("Generating observations...")
start_time = time.perf_counter()
true_mean = compute_piecewise_mean(
    x, true_slopes, true_biases, true_switch_indices
)
observations = true_mean + sigma * jax.random.normal(
    noise_key, (num_observations,)
)
print(f"observations build: {time.perf_counter() - start_time:.3f}s")

print("Building switch table...")
start_time = time.perf_counter()
switch_table = build_switch_table(num_observations, num_switches)
print(
    f"switch table build: {time.perf_counter() - start_time:.3f}s "
    f"(configs={switch_table.shape[0]})"
)

print("Building model...")
start_time = time.perf_counter()
model = build_switch_model(x, sigma, switch_table, num_switches)
model_marg = paz.marginalize(model, ["switch_index"])
data = {"y": observations}
print(f"model build: {time.perf_counter() - start_time:.3f}s")

result = run_inference(
    model_marg,
    key,
    num_switches,
    switch_table,
    data,
    num_samples,
    num_chains,
    burn_in,
    step_sigma,
)

print("Computing switch posterior summary...")
start_time = time.perf_counter()
num_positions = num_observations - 1
switch_positions = 0.5 * (x[:-1] + x[1:])
switch_position_posterior = compute_switch_position_posterior(
    result.switch_table, result.posterior_configs, num_positions
)
map_config_index = int(jp.argmax(result.posterior_configs))
map_switch_indices = result.switch_table[map_config_index]
print(f"switch posterior summary: {time.perf_counter() - start_time:.3f}s")

print("Computing posterior mean line...")
start_time = time.perf_counter()
posterior_mean = compute_piecewise_mean(
    x, result.slope_means, result.bias_means, map_switch_indices
)
print(f"posterior mean build: {time.perf_counter() - start_time:.3f}s")

print("Computing gaussian approximation line...")
start_time = time.perf_counter()
density_slopes, density_biases = extract_segment_values(
    result.density_sample, num_switches
)
density_mean = compute_piecewise_mean(
    x, density_slopes, density_biases, map_switch_indices
)
print(f"gaussian mean build: {time.perf_counter() - start_time:.3f}s")

print("=" * 60)
print("Multi change-point regression")
print("=" * 60)
print(f"num_switches={num_switches}")
print(f"num_configs={result.switch_table.shape[0]}")
print(f"true switch indices={true_switch_indices.tolist()}")
print(f"map switch indices={map_switch_indices.tolist()}")
print(f"acceptance rate={result.acceptance_rate:.3f}")
print(f"mcmc seconds={result.mcmc_seconds:.2f}")
print(f"posterior seconds={result.posterior_seconds:.2f}")

print("Plotting...")
start_time = time.perf_counter()
plot_results(
    x,
    observations,
    posterior_mean,
    density_mean,
    true_mean,
    switch_positions,
    map_switch_indices,
    true_switch_indices,
    switch_position_posterior,
)
print(f"plotting: {time.perf_counter() - start_time:.3f}s")
plot.show()

