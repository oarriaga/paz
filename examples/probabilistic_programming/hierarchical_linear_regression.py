"""
Hierarchical Bayesian Linear Regression

Demonstrates hierarchical modeling with group-level varying slopes and intercepts.
Supports both CENTERED and NON-CENTERED parameterizations.

PARAMETERIZATION FLAG:
  Set PARAMETERIZATION = "centered" or "non-centered" (line ~184)

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

from collections import namedtuple

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz
import paz.plot as plot

tfd = tfp.distributions
tfb = tfp.bijectors

TrueParams = namedtuple(
    "TrueParams",
    ["mu_slope", "sigma_slope", "mu_intercept", "sigma_intercept", "sigma_obs"],
)


def generate_hierarchical_data(key, num_groups, num_per_group, true_params):
    mu_slope, sigma_slope = true_params.mu_slope, true_params.sigma_slope
    mu_intercept, sigma_intercept = (
        true_params.mu_intercept,
        true_params.sigma_intercept,
    )
    sigma_obs = true_params.sigma_obs

    data_key, slopes_key, intercepts_key = jax.random.split(key, 3)
    true_slopes = (
        jax.random.normal(slopes_key, (num_groups,)) * sigma_slope + mu_slope
    )
    true_intercepts = (
        jax.random.normal(intercepts_key, (num_groups,)) * sigma_intercept
        + mu_intercept
    )

    X, y, group_idx = [], [], []
    for j in range(num_groups):
        data_key, x_key, noise_key = jax.random.split(data_key, 3)
        x_j = jax.random.uniform(x_key, (num_per_group,), minval=0, maxval=1)
        noise = jax.random.normal(noise_key, (num_per_group,)) * sigma_obs
        y_j = true_slopes[j] * x_j + true_intercepts[j] + noise
        X.append(x_j)
        y.append(y_j)
        group_idx.append(jp.full(num_per_group, j, dtype=jp.int32))

    X = jp.concatenate(X)
    y = jp.concatenate(y)
    group_idx = jp.concatenate(group_idx)
    return X, y, group_idx, true_slopes, true_intercepts


def HierarchicalLikelihood(X, group_idx):
    def apply(slopes, intercepts, sigma_obs):
        means = slopes[group_idx] * X + intercepts[group_idx]
        return tfd.Normal(means, sigma_obs)

    return apply


def SlopePrior(num_groups):
    def apply(mu_slope, sigma_slope):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_slope), sigma_slope),
            reinterpreted_batch_ndims=1,
        )

    return apply


def InterceptPrior(num_groups):
    def apply(mu_intercept, sigma_intercept):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_intercept), sigma_intercept),
            reinterpreted_batch_ndims=1,
        )

    return apply


def build_centered_model(
    X, group_idx, num_groups, sigma_bijector, obs_bijector
):
    """
    Centered parameterization (standard hierarchical model).
    slopes ~ Normal(mu_slope, sigma_slope)
    intercepts ~ Normal(mu_intercept, sigma_intercept)
    """
    mu_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_slope")
    mu_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_intercept")

    sigma_slope = paz.Prior(
        tfd.Uniform(0.01, 1.0), name="sigma_slope", bijector=sigma_bijector
    )
    sigma_intercept = paz.Prior(
        tfd.Uniform(0.01, 1.0),
        name="sigma_intercept",
        bijector=sigma_bijector,
    )

    slopes = paz.Latent(SlopePrior(num_groups), name="slopes")(
        mu_slope, sigma_slope
    )
    intercepts = paz.Latent(InterceptPrior(num_groups), name="intercepts")(
        mu_intercept, sigma_intercept
    )

    sigma_obs = paz.Prior(
        tfd.Uniform(0.01, 0.5), name="sigma_obs", bijector=obs_bijector
    )

    y_obs = paz.Observable(HierarchicalLikelihood(X, group_idx), name="y_obs")(
        slopes, intercepts, sigma_obs
    )

    priors = [mu_slope, mu_intercept, sigma_slope, sigma_intercept, sigma_obs]
    return paz.PGM(priors, [y_obs], "hierarchical_centered")


def build_noncentered_model(
    X, group_idx, num_groups, sigma_bijector, obs_bijector
):
    """
    Non-centered parameterization (avoids Neal's funnel).
    z_slopes ~ Normal(0, 1)
    slopes = mu_slope + sigma_slope * z_slopes
    z_intercepts ~ Normal(0, 1)
    intercepts = mu_intercept + sigma_intercept * z_intercepts
    """
    mu_slope = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_slope")
    mu_intercept = paz.Prior(tfd.Normal(0.0, 1.0), name="mu_intercept")

    sigma_slope = paz.Prior(
        tfd.Uniform(0.01, 1.0), name="sigma_slope", bijector=sigma_bijector
    )
    sigma_intercept = paz.Prior(
        tfd.Uniform(0.01, 1.0),
        name="sigma_intercept",
        bijector=sigma_bijector,
    )

    # Non-centered: sample standard normal offsets
    z_slopes = paz.Prior(
        tfd.Independent(
            tfd.Normal(jp.zeros(num_groups), 1.0), reinterpreted_batch_ndims=1
        ),
        name="z_slopes",
    )
    z_intercepts = paz.Prior(
        tfd.Independent(
            tfd.Normal(jp.zeros(num_groups), 1.0), reinterpreted_batch_ndims=1
        ),
        name="z_intercepts",
    )

    sigma_obs = paz.Prior(
        tfd.Uniform(0.01, 0.5), name="sigma_obs", bijector=obs_bijector
    )

    # Reparameterization happens inside likelihood
    def likelihood_noncentered(
        z_slopes,
        z_intercepts,
        mu_slope,
        sigma_slope,
        mu_intercept,
        sigma_intercept,
        sigma_obs,
    ):
        slopes = mu_slope + sigma_slope * z_slopes
        intercepts = mu_intercept + sigma_intercept * z_intercepts
        means = slopes[group_idx] * X + intercepts[group_idx]
        return tfd.Normal(means, sigma_obs)

    y_obs = paz.Observable(likelihood_noncentered, name="y_obs")(
        z_slopes,
        z_intercepts,
        mu_slope,
        sigma_slope,
        mu_intercept,
        sigma_intercept,
        sigma_obs,
    )

    priors = [
        mu_slope,
        mu_intercept,
        sigma_slope,
        sigma_intercept,
        sigma_obs,
        z_slopes,
        z_intercepts,
    ]
    return paz.PGM(priors, [y_obs], "hierarchical_noncentered")


# Configure plotting
plot.configure(fontsize=12, latex=False)

true_params = TrueParams(
    mu_slope=0.8,
    sigma_slope=0.3,
    mu_intercept=0.2,
    sigma_intercept=0.2,
    sigma_obs=0.1,
)

num_groups = 5
num_per_group = 50
PLOT = True
PARAMETERIZATION = "non-centered"  # "centered" or "non-centered"

# Get group colors
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(num_groups)]

key = jax.random.PRNGKey(7)
key, data_key = jax.random.split(key)
X, y, group_idx, true_slopes, true_intercepts = generate_hierarchical_data(
    data_key, num_groups, num_per_group, true_params
)
print("X.shape", X.shape)
data = {"y_obs": y}

print(f"Parameterization: {PARAMETERIZATION.upper()}")
print("True parameters:")
print(
    f"  mu_slope: {true_params.mu_slope}, "
    f"sigma_slope: {true_params.sigma_slope}"
)
print(
    f"  mu_intercept: {true_params.mu_intercept}, "
    f"sigma_intercept: {true_params.sigma_intercept}"
)
print(f"  sigma_obs: {true_params.sigma_obs}")
print(f"  true_slopes: {true_slopes}")
print(f"  true_intercepts: {true_intercepts}")

# Define bijectors for constrained parameters
low, high = 0.01, 1.0
sigma_bijector = tfb.Chain(
    [tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()]
)
obs_low, obs_high = 0.01, 0.5
obs_bijector = tfb.Chain(
    [tfb.Shift(obs_low), tfb.Scale(obs_high - obs_low), tfb.Sigmoid()]
)

# Build model based on parameterization flag
if PARAMETERIZATION == "centered":
    model = build_centered_model(
        X, group_idx, num_groups, sigma_bijector, obs_bijector
    )
elif PARAMETERIZATION == "non-centered":
    model = build_noncentered_model(
        X, group_idx, num_groups, sigma_bijector, obs_bijector
    )
else:
    raise ValueError(f"Unknown parameterization: {PARAMETERIZATION}")

print("\nSampling from prior predictive...")
key, prior_key = jax.random.split(key)

if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Prior predictive
    for key_i in jax.random.split(prior_key, 20):
        sample = model.sample(key_i)
        if PARAMETERIZATION == "centered":
            slopes_sample = sample.slopes
            intercepts_sample = sample.intercepts
        else:  # non-centered
            slopes_sample = (
                sample.mu_slope + sample.sigma_slope * sample.z_slopes
            )
            intercepts_sample = (
                sample.mu_intercept
                + sample.sigma_intercept * sample.z_intercepts
            )

        for j in range(num_groups):
            mask = group_idx == j
            x_j = X[mask]
            sort_idx = jp.argsort(x_j)
            y_pred = slopes_sample[j] * x_j + intercepts_sample[j]
            axes[0].plot(
                np.array(x_j[sort_idx]), np.array(y_pred[sort_idx]),
                color=colors[j], alpha=0.1
            )
    axes[0].set_title("Prior predictive")
    plot.set_labels(axes[0], x="X", y="y")
    plot.clean(axes[0])

    # Observed data with groups
    for j in range(num_groups):
        mask = group_idx == j
        plot.scatter(np.array(X[mask]), np.array(y[mask]), axes[1],
                     s=20, alpha=0.6, color=colors[j])
    axes[1].set_title("Observed data")
    plot.set_labels(axes[1], x="X", y="y")
    plot.clean(axes[1])

    plt.tight_layout()
    plot.show()

num_chains = 10
num_samples = 70_000
sigma = 0.02
burn_in = 0.20

print(f"\nRunning MCMC with {num_samples} samples, {num_chains} chains...")
key, mcmc_key = jax.random.split(key, 2)
model.configure(
    num_chains=num_chains,
    warmup=burn_in,
    sigma=sigma,
    tuner=paz.AdaptiveStepTuner(sigma=sigma),
)
posterior = model.infer(mcmc_key, data, num_samples=num_samples)
samples, infos = posterior.inverse_samples, posterior.infos
posterior_forward = posterior.samples

print(f"\nMean acceptance rate: {infos.acceptance_rate.mean():.3f}")

print("\nPosterior estimates vs true values:")
print(
    f"  mu_slope: {samples.mu_slope.mean():.3f} "
    f"(true: {true_params.mu_slope})"
)
print(
    f"  mu_intercept: {samples.mu_intercept.mean():.3f} "
    f"(true: {true_params.mu_intercept})"
)
print(
    f"  sigma_slope: "
    f"{posterior_forward.sigma_slope.mean():.3f} "
    f"(true: {true_params.sigma_slope})"
)
print(
    f"  sigma_intercept: "
    f"{posterior_forward.sigma_intercept.mean():.3f} "
    f"(true: {true_params.sigma_intercept})"
)
print(
    f"  sigma_obs: {posterior_forward.sigma_obs.mean():.3f} "
    f"(true: {true_params.sigma_obs})"
)

# Extract slopes and intercepts (handle both parameterizations)
if PARAMETERIZATION == "centered":
    posterior_slopes = samples.slopes.reshape(-1, num_groups)
    posterior_intercepts = samples.intercepts.reshape(-1, num_groups)
else:  # non-centered
    mu_slope_expanded = samples.mu_slope.reshape(-1, 1)
    sigma_slope_expanded = posterior_forward.sigma_slope.reshape(-1, 1)
    z_slopes = samples.z_slopes.reshape(-1, num_groups)
    posterior_slopes = mu_slope_expanded + sigma_slope_expanded * z_slopes

    mu_intercept_expanded = samples.mu_intercept.reshape(-1, 1)
    sigma_intercept_expanded = posterior_forward.sigma_intercept.reshape(-1, 1)
    z_intercepts = samples.z_intercepts.reshape(-1, num_groups)
    posterior_intercepts = (
        mu_intercept_expanded + sigma_intercept_expanded * z_intercepts
    )

density = posterior.as_density(method="gaussian")
key, density_key = jax.random.split(key)
density_forward = density.sample(density_key, num_samples=1)
if PARAMETERIZATION == "centered":
    density_slopes = density_forward.slopes
    density_intercepts = density_forward.intercepts
else:
    density_slopes = (
        density_forward.mu_slope
        + density_forward.sigma_slope * density_forward.z_slopes
    )
    density_intercepts = (
        density_forward.mu_intercept
        + density_forward.sigma_intercept * density_forward.z_intercepts
    )

print("\nGroup-level estimates:")
for j in range(num_groups):
    print(
        f"  Group {j}: slope={posterior_slopes[:, j].mean():.3f} "
        f"(true: {true_slopes[j]:.3f}), "
        f"intercept={posterior_intercepts[:, j].mean():.3f} "
        f"(true: {true_intercepts[j]:.3f})"
    )

if PLOT:
    # Posterior panel for hyperparameters
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    plot.histogram(np.array(samples.mu_slope.flatten()), axes[0, 0],
                   bins=50, alpha=0.7, color=plot.BLUE_GREY.primary)
    plot.vline(true_params.mu_slope, axes[0, 0], color=plot.EARTH.primary,
               linestyle="--", label="True")
    plot.set_labels(axes[0, 0], x="mu_slope", y="density")
    axes[0, 0].legend()
    plot.clean(axes[0, 0])

    plot.histogram(np.array(samples.mu_intercept.flatten()), axes[0, 1],
                   bins=50, alpha=0.7, color=plot.BLUE_GREY.primary)
    plot.vline(true_params.mu_intercept, axes[0, 1], color=plot.EARTH.primary,
               linestyle="--", label="True")
    plot.set_labels(axes[0, 1], x="mu_intercept", y="density")
    axes[0, 1].legend()
    plot.clean(axes[0, 1])

    plot.histogram(
        np.array(posterior_forward.sigma_slope.flatten()),
        axes[0, 2],
        bins=50,
        alpha=0.7,
        color=plot.BLUE_GREY.primary,
    )
    plot.vline(true_params.sigma_slope, axes[0, 2], color=plot.EARTH.primary,
               linestyle="--", label="True")
    plot.set_labels(axes[0, 2], x="sigma_slope", y="density")
    axes[0, 2].legend()
    plot.clean(axes[0, 2])

    plot.histogram(
        np.array(posterior_forward.sigma_intercept.flatten()),
        axes[1, 0],
        bins=50,
        alpha=0.7,
        color=plot.BLUE_GREY.primary,
    )
    plot.vline(
        true_params.sigma_intercept,
        axes[1, 0],
        color=plot.EARTH.primary,
        linestyle="--",
        label="True",
    )
    plot.set_labels(axes[1, 0], x="sigma_intercept", y="density")
    axes[1, 0].legend()
    plot.clean(axes[1, 0])

    plot.histogram(np.array(posterior_forward.sigma_obs.flatten()), axes[1, 1],
                   bins=50, alpha=0.7, color=plot.BLUE_GREY.primary)
    plot.vline(true_params.sigma_obs, axes[1, 1], color=plot.EARTH.primary,
               linestyle="--", label="True")
    plot.set_labels(axes[1, 1], x="sigma_obs", y="density")
    axes[1, 1].legend()
    plot.clean(axes[1, 1])

    # Group parameter scatter (slope vs intercept)
    for j in range(num_groups):
        axes[1, 2].scatter(
            float(posterior_slopes[:, j].mean()),
            float(posterior_intercepts[:, j].mean()),
            color=colors[j], s=100, label=f"Group {j} (post)", zorder=3,
        )
        axes[1, 2].scatter(
            float(true_slopes[j]), float(true_intercepts[j]),
            color=colors[j], marker="x", s=100, zorder=3,
        )
        axes[1, 2].scatter(
            float(density_slopes[j]), float(density_intercepts[j]),
            color="tab:purple", marker="D", s=80, zorder=4,
        )
    axes[1, 2].scatter([], [], color="tab:purple", marker="D", s=80,
                       label="gaussian approx")
    plot.set_labels(axes[1, 2], x="slope", y="intercept")
    axes[1, 2].set_title("Group parameters (circle=posterior, x=true)")
    axes[1, 2].legend(loc="upper left", fontsize=8)
    plot.clean(axes[1, 2])

    plt.tight_layout()
    plot.show()

    # Posterior predictive by group
    fig, axes = plt.subplots(1, num_groups, figsize=(12, 4))
    for j in range(num_groups):
        mask = group_idx == j
        plot.scatter(np.array(X[mask]), np.array(y[mask]), axes[j],
                     s=20, alpha=0.6, color=colors[j])

        x_plot = np.linspace(0, 1, 100)
        for i in range(50):
            idx = int(jax.random.randint(
                jax.random.PRNGKey(i), (), 0, len(posterior_slopes)
            ))
            slope_i = float(posterior_slopes[idx, j])
            intercept_i = float(posterior_intercepts[idx, j])
            plot.line(x_plot, slope_i * x_plot + intercept_i, axes[j],
                      color=colors[j], alpha=0.1)

        plot.line(
            x_plot,
            float(true_slopes[j]) * x_plot + float(true_intercepts[j]),
            axes[j],
            color="black",
            linestyle="--",
            linewidth=2,
            label="True",
        )
        axes[j].set_title(f"Group {j}")
        plot.set_labels(axes[j], x="X", y="y" if j == 0 else None)
        if j == 0:
            axes[j].legend()
        plot.clean(axes[j])

    plt.suptitle("Posterior predictive by group", y=1.02)
    plt.tight_layout()
    plot.show()

    # Diagnostics
    fig, ax = plot.subplots()
    plot.diagnostics(infos.acceptance_rate, ax, color=plot.DANDELION.primary)
    ax.set_title("Acceptance rates per chain")
    plot.clean(ax)
    plot.show()

    # Corner plot for hyperparameters
    plot.corner({
        "mu_slope": np.array(samples.mu_slope.flatten()),
        "mu_intercept": np.array(samples.mu_intercept.flatten()),
    }, true_values={
        "mu_slope": true_params.mu_slope,
        "mu_intercept": true_params.mu_intercept,
    })
    plot.show()

    # Forest plot for group slopes
    group_names = [f"slope_{j}" for j in range(num_groups)]
    group_means = [
        float(posterior_slopes[:, j].mean()) for j in range(num_groups)
    ]
    group_errors = [
        float(posterior_slopes[:, j].std()) for j in range(num_groups)
    ]
    group_true = [float(true_slopes[j]) for j in range(num_groups)]

    fig, ax = plot.subplots(figsize=(8, 5))
    plot.forest_plot(group_names, group_means, group_errors, ax,
                     true_values=group_true, color=plot.BLUE_GREY.primary)
    ax.set_title("Group slope estimates (with 1 std)")
    plot.clean(ax, spines="box")
    plot.show()

print("\nShrinkage effect (group estimates pulled toward global mean):")
pooled_slope = samples.mu_slope.mean()
pooled_intercept = samples.mu_intercept.mean()
print(f"  Pooled slope mean: {pooled_slope:.3f}")
print(f"  Pooled intercept mean: {pooled_intercept:.3f}")


