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
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz

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


def main():
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
    # PARAMETERIZATION = "centered"  # "centered" or "non-centered"
    if PLOT:
        colors = plt.cm.tab10(jp.arange(num_groups))
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
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        for key_i in jax.random.split(prior_key, 20):
            sample = model.sample(key_i)
            # Extract slopes and intercepts (handle both parameterizations)
            if PARAMETERIZATION == "centered":
                slopes_sample = sample.slopes
                intercepts_sample = sample.intercepts
            else:  # non-centered
                # Transform z to actual parameters
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
                plt.plot(
                    x_j[sort_idx], y_pred[sort_idx], color=colors[j], alpha=0.1
                )
        plt.title("Prior predictive")
        plt.xlabel("X")
        plt.ylabel("y")

        plt.subplot(1, 2, 2)
        for j in range(num_groups):
            mask = group_idx == j
            plt.scatter(
                X[mask],
                y[mask],
                color=colors[j],
                alpha=0.6,
                label=f"Group {j}",
            )
        plt.legend()
        plt.title("Observed data")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()

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
        f"  mu_slope: {samples.position.mu_slope.mean():.3f} "
        f"(true: {true_params.mu_slope})"
    )
    print(
        f"  mu_intercept: {samples.position.mu_intercept.mean():.3f} "
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
        posterior_slopes = samples.position.slopes.reshape(-1, num_groups)
        posterior_intercepts = samples.position.intercepts.reshape(
            -1, num_groups
        )
    else:  # non-centered
        # Transform z to actual parameters
        mu_slope_expanded = samples.position.mu_slope.reshape(-1, 1)
        sigma_slope_expanded = posterior_forward.sigma_slope.reshape(-1, 1)
        z_slopes = samples.position.z_slopes.reshape(-1, num_groups)
        posterior_slopes = mu_slope_expanded + sigma_slope_expanded * z_slopes

        mu_intercept_expanded = samples.position.mu_intercept.reshape(-1, 1)
        sigma_intercept_expanded = posterior_forward.sigma_intercept.reshape(
            -1, 1
        )
        z_intercepts = samples.position.z_intercepts.reshape(-1, num_groups)
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
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        axes[0, 0].hist(
            samples.position.mu_slope.flatten(),
            bins=50,
            density=True,
            alpha=0.7,
        )
        axes[0, 0].axvline(
            true_params.mu_slope, color="red", linestyle="--", label="True"
        )
        axes[0, 0].set_xlabel("mu_slope")
        axes[0, 0].legend()

        axes[0, 1].hist(
            samples.position.mu_intercept.flatten(),
            bins=50,
            density=True,
            alpha=0.7,
        )
        axes[0, 1].axvline(
            true_params.mu_intercept, color="red", linestyle="--", label="True"
        )
        axes[0, 1].set_xlabel("mu_intercept")
        axes[0, 1].legend()

        axes[0, 2].hist(
            posterior_forward.sigma_slope.flatten(),
            bins=50,
            density=True,
            alpha=0.7,
        )
        axes[0, 2].axvline(
            true_params.sigma_slope, color="red", linestyle="--", label="True"
        )
        axes[0, 2].set_xlabel("sigma_slope")
        axes[0, 2].legend()

        axes[1, 0].hist(
            posterior_forward.sigma_intercept.flatten(),
            bins=50,
            density=True,
            alpha=0.7,
        )
        axes[1, 0].axvline(
            true_params.sigma_intercept,
            color="red",
            linestyle="--",
            label="True",
        )
        axes[1, 0].set_xlabel("sigma_intercept")
        axes[1, 0].legend()

        axes[1, 1].hist(
            posterior_forward.sigma_obs.flatten(),
            bins=50,
            density=True,
            alpha=0.7,
        )
        axes[1, 1].axvline(
            true_params.sigma_obs, color="red", linestyle="--", label="True"
        )
        axes[1, 1].set_xlabel("sigma_obs")
        axes[1, 1].legend()

        for j in range(num_groups):
            axes[1, 2].scatter(
                posterior_slopes[:, j].mean(),
                posterior_intercepts[:, j].mean(),
                color=colors[j],
                s=100,
                label=f"Group {j} (post)",
                zorder=3,
            )
            axes[1, 2].scatter(
                true_slopes[j],
                true_intercepts[j],
                color=colors[j],
                marker="x",
                s=100,
                zorder=3,
            )
            axes[1, 2].scatter(
                density_slopes[j],
                density_intercepts[j],
                color="tab:purple",
                marker="D",
                s=80,
                zorder=4,
            )
        axes[1, 2].scatter(
            [],
            [],
            color="tab:purple",
            marker="D",
            s=80,
            label="gaussian approx",
        )

        axes[1, 2].set_xlabel("slope")
        axes[1, 2].set_ylabel("intercept")
        axes[1, 2].set_title("Group parameters (circle=posterior, x=true)")
        axes[1, 2].legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        for j in range(num_groups):
            plt.subplot(1, num_groups, j + 1)
            mask = group_idx == j
            plt.scatter(X[mask], y[mask], color=colors[j], alpha=0.6, s=20)

            x_plot = jp.linspace(0, 1, 100)
            for i in range(50):
                idx = jax.random.randint(
                    jax.random.PRNGKey(i), (), 0, len(posterior_slopes)
                )
                slope_i = posterior_slopes[idx, j]
                intercept_i = posterior_intercepts[idx, j]
                plt.plot(
                    x_plot,
                    slope_i * x_plot + intercept_i,
                    color=colors[j],
                    alpha=0.1,
                )

            plt.plot(
                x_plot,
                true_slopes[j] * x_plot + true_intercepts[j],
                "k--",
                linewidth=2,
                label="True",
            )
            plt.title(f"Group {j}")
            plt.xlabel("X")
            if j == 0:
                plt.ylabel("y")
            plt.legend()

        plt.suptitle("Posterior predictive by group", y=1.02)
        plt.tight_layout()
        plt.show()

    print("\nShrinkage effect (group estimates pulled toward global mean):")
    pooled_slope = samples.position.mu_slope.mean()
    pooled_intercept = samples.position.mu_intercept.mean()
    print(f"  Pooled slope mean: {pooled_slope:.3f}")
    print(f"  Pooled intercept mean: {pooled_intercept:.3f}")


if __name__ == "__main__":
    main()
