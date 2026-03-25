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

import time
from collections import namedtuple

import jax
import jax.numpy as jp
from matplotlib.patches import Rectangle
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.utils.plot as plot
from paz.inference.types import SampleType

tfd = tfp.distributions
tfb = tfp.bijectors
CaseResult = namedtuple(
    "CaseResult",
    [
        "observations",
        "slope_mean",
        "bias_mean",
        "posterior_z",
        "acceptance_rate",
        "mcmc_seconds",
        "posterior_seconds",
        "z_true",
    ],
)


def build_switch_model(x, sigma_in, sigma_out):
    slope = paz.Prior(tfd.Normal(0.0, 1.0), name="slope")
    bias = paz.Prior(tfd.Normal(0.0, 1.0), name="bias")
    p = paz.Prior(tfd.Beta(2.0, 2.0), name="p", bijector=tfb.Sigmoid())

    def z_distribution(p):
        return tfd.Bernoulli(probs=p, dtype=jp.float32)

    z = paz.Latent(z_distribution, name="z")(p)

    def y_distribution(slope, bias, z):
        mean = slope * x + bias
        sigma = jp.where(z == 1.0, sigma_out, sigma_in)
        return tfd.Normal(mean, sigma)

    y_obs = paz.Observable(y_distribution, name="y")(slope, bias, z)
    return paz.PGM([slope, bias, p], [y_obs], "robust_outlier_switch")


def run_mcmc(model_marg, key, data, num_samples, num_chains, sigma, warmup):
    model_marg.configure(
        num_chains=num_chains,
        warmup=warmup,
        sigma=sigma,
        tuner=paz.AdaptiveStepTuner(sigma=sigma),
    )
    return model_marg.infer(key, data, num_samples=num_samples)


def plot_switch_results(
    axes,
    x,
    observations,
    slope_mean,
    bias_mean,
    sigma_in,
    sigma_out,
    posterior_z,
    slope_true,
    bias_true,
    z_true,
    density_params=None,
):
    axis_top, axis_bottom = axes
    x_grid = jp.linspace(x.min(), x.max(), 200)
    mean_line = slope_mean * x_grid + bias_mean
    in_upper = mean_line + 2.0 * sigma_in
    in_lower = mean_line - 2.0 * sigma_in
    out_upper = mean_line + 2.0 * sigma_out
    out_lower = mean_line - 2.0 * sigma_out

    highlight_outlier = posterior_z[1] >= posterior_z[0]
    if highlight_outlier:
        selected_upper, selected_lower = out_upper, out_lower
        alt_upper, alt_lower = in_upper, in_lower
        selected_color, alt_color = "tab:orange", "tab:blue"
        selected_label = f"selected band (sigma={sigma_out})"
        alt_label = f"alt band (sigma={sigma_in})"
    else:
        selected_upper, selected_lower = in_upper, in_lower
        alt_upper, alt_lower = out_upper, out_lower
        selected_color, alt_color = "tab:blue", "tab:orange"
        selected_label = f"selected band (sigma={sigma_in})"
        alt_label = f"alt band (sigma={sigma_out})"

    plot.scatter(x, observations, axis_top, color="tab:gray", alpha=0.8,
                 label="data")
    plot.line(x_grid, mean_line, axis_top, color="black",
              label="posterior mean line")
    if density_params is not None:
        density_slope, density_bias = density_params
        density_line = density_slope * x_grid + density_bias
        plot.line(x_grid, density_line, axis_top, color="tab:purple",
                  linestyle="--", label="gaussian approx")
    true_line = slope_true * x_grid + bias_true
    plot.line(x_grid, true_line, axis_top, color="tab:green", linestyle="--",
              label="true line")
    plot.fill_between(x_grid, selected_lower, selected_upper, axis_top,
                      color=selected_color, alpha=0.25, label=selected_label)
    plot.line(x_grid, alt_upper, axis_top, color=alt_color, linestyle=":",
              linewidth=1.5, label=alt_label)
    plot.line(x_grid, alt_lower, axis_top, color=alt_color, linestyle=":",
              linewidth=1.5)
    switch_label = "outlier" if highlight_outlier else "inlier"
    axis_top.set_title(
        f"Global switch selects: {switch_label} "
        f"(p(z=1|y)={posterior_z[1]:.2f}, z_true={int(z_true)})"
    )
    axis_top.add_patch(
        Rectangle(
            (0.965, 0.1),
            0.02,
            0.8,
            transform=axis_top.transAxes,
            color=selected_color,
            alpha=0.6,
            clip_on=False,
        )
    )
    axis_top.text(
        0.975,
        0.92,
        "selected",
        transform=axis_top.transAxes,
        rotation=90,
        ha="center",
        va="top",
        color=selected_color,
        fontsize=9,
    )
    plot.legend(axis_top, loc="upper left")
    plot.clean(axis_top)

    axis_bottom.bar(
        [0, 1],
        posterior_z,
        color=["tab:blue", "tab:orange"],
        tick_label=["z=0 (inlier)", "z=1 (outlier)"],
    )
    axis_bottom.set_ylim(0.0, 1.0)
    axis_bottom.set_ylabel("Posterior probability")
    plot.clean(axis_bottom)
    axis_bottom.set_title("Posterior over switch z")


def run_case(
    key,
    x,
    observations,
    sigma_in,
    sigma_out,
    z_true,
):
    model = build_switch_model(x, sigma_in, sigma_out)
    model_marg = paz.marginalize(model, ["z"])
    data = {"y": observations}

    num_samples = 600
    num_chains = 2
    burn_in = 150
    sigma = 0.25

    start_time = time.perf_counter()
    posterior = run_mcmc(
        model_marg, key, data, num_samples, num_chains, sigma, burn_in
    )
    samples, infos = posterior.inverse_samples, posterior.infos
    mcmc_seconds = time.perf_counter() - start_time
    slope_samples = samples.position.slope.reshape(-1)
    bias_samples = samples.position.bias.reshape(-1)
    p_inverse_samples = samples.position.p.reshape(-1)

    slope_mean = slope_samples.mean()
    bias_mean = bias_samples.mean()
    acceptance_rate = infos.acceptance_rate.mean()

    start_time = time.perf_counter()
    Theta = SampleType(["slope", "bias", "p"])
    theta_samples = Theta(slope_samples, bias_samples, p_inverse_samples)
    posterior_z = paz.recover_discrete_posterior(
        model_marg, "z", theta_samples, data
    ).posterior.mean(axis=0)
    posterior_seconds = time.perf_counter() - start_time

    density = posterior.as_density(method="gaussian")
    key, density_key = jax.random.split(key)
    density_sample = density.sample(density_key, num_samples=1)
    density_params = (density_sample.slope, density_sample.bias)

    return (
        CaseResult(
            observations,
            slope_mean,
            bias_mean,
            posterior_z,
            acceptance_rate,
            mcmc_seconds,
            posterior_seconds,
            z_true,
        ),
        density_params,
        key,
    )


key = jax.random.PRNGKey(7)
num_observations = 30
x = jp.linspace(-1.0, 1.0, num_observations)
slope_true = 1.2
bias_true = -0.2
sigma_in = 0.2
sigma_out = 1.0

case_results = []
for z_value in [0.0, 1.0]:
    key, noise_key = jax.random.split(key)
    sigma_true = jp.where(z_value == 1.0, sigma_out, sigma_in)
    observations = (
        slope_true * x
        + bias_true
        + sigma_true * jax.random.normal(noise_key, (num_observations,))
    )
    result, density_params, key = run_case(
        key,
        x,
        observations,
        sigma_in,
        sigma_out,
        jp.array(z_value),
    )
    case_results.append((result, density_params))

    print("=" * 60)
    print("Robust linear regression with outlier switch")
    print("=" * 60)
    print(
        f"true slope={slope_true}, true bias={bias_true}, "
        f"z_true={int(z_value)}"
    )
    print(
        f"posterior mean slope={result.slope_mean:.3f}, "
        f"bias={result.bias_mean:.3f}"
    )
    print(f"posterior p(z=1 | y)={result.posterior_z[1]:.3f}")
    print(f"acceptance rate={result.acceptance_rate:.3f}")
    print(f"mcmc seconds={result.mcmc_seconds:.2f}")
    print(f"posterior seconds={result.posterior_seconds:.2f}")

plot.configure(fontsize=12)
figure, axes = plot.subplots(nrows=2, ncols=2, figsize=(14, 8))
for column, result in enumerate(case_results):
    case_result, density_params = result
    plot_switch_results(
        (axes[0, column], axes[1, column]),
        x,
        case_result.observations,
        case_result.slope_mean,
        case_result.bias_mean,
        sigma_in,
        sigma_out,
        case_result.posterior_z,
        slope_true,
        bias_true,
        case_result.z_true,
        density_params,
    )
figure.tight_layout()
plot.show()
