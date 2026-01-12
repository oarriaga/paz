from collections import namedtuple

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import paz
tfd = tfp.distributions
tfb = tfp.bijectors


def generate_hierarchical_data(key, num_groups, num_per_group, true_params):
    mu_slope, sigma_slope = true_params["mu_slope"], true_params["sigma_slope"]
    mu_intercept, sigma_intercept = true_params["mu_intercept"], true_params["sigma_intercept"]
    sigma_obs = true_params["sigma_obs"]

    data_key, slopes_key, intercepts_key = jax.random.split(key, 3)
    true_slopes = jax.random.normal(slopes_key, (num_groups,)) * sigma_slope + mu_slope
    true_intercepts = jax.random.normal(intercepts_key, (num_groups,)) * sigma_intercept + mu_intercept

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
            reinterpreted_batch_ndims=1
        )
    return apply


def InterceptPrior(num_groups):
    def apply(mu_intercept, sigma_intercept):
        return tfd.Independent(
            tfd.Normal(jp.full(num_groups, mu_intercept), sigma_intercept),
            reinterpreted_batch_ndims=1
        )
    return apply


Parameters = namedtuple(
    "Parameters",
    [
        "mu_slope",
        "mu_intercept",
        "sigma_slope",
        "sigma_intercept",
        "sigma_obs",
        "slopes",
        "intercepts",
    ],
)


def apply_prior(prior, inverse_sample):
    state = prior.apply(inverse_sample)
    return getattr(state.sample, prior.name), state.log_prob


def apply_latent(latent, inverse_sample, *args):
    state = latent.apply(inverse_sample, *args)
    return getattr(state.sample, latent.name), state.log_prob


def sample_initial_positions(
    key,
    num_samples,
    mu_slope,
    mu_intercept,
    sigma_slope,
    sigma_intercept,
    sigma_obs,
    slopes,
    intercepts,
):
    keys = jax.random.split(key, 7)
    mu_slope_inverse = mu_slope.sample_inverse(keys[0], num_samples)
    mu_intercept_inverse = mu_intercept.sample_inverse(keys[1], num_samples)
    sigma_slope_inverse = sigma_slope.sample_inverse(keys[2], num_samples)
    sigma_intercept_inverse = sigma_intercept.sample_inverse(keys[3], num_samples)
    sigma_obs_inverse = sigma_obs.sample_inverse(keys[4], num_samples)

    mu_slope_forward, _ = apply_prior(mu_slope, mu_slope_inverse)
    mu_intercept_forward, _ = apply_prior(mu_intercept, mu_intercept_inverse)
    sigma_slope_forward, _ = apply_prior(sigma_slope, sigma_slope_inverse)
    sigma_intercept_forward, _ = apply_prior(sigma_intercept, sigma_intercept_inverse)

    def sample_latent_inverse(latent, key, mu_value, sigma_value):
        return latent.sample_inverse(key, 1, mu_value, sigma_value)

    slope_keys = jax.random.split(keys[5], num_samples)
    intercept_keys = jax.random.split(keys[6], num_samples)
    slopes_inverse = jax.vmap(sample_latent_inverse, in_axes=(None, 0, 0, 0))(
        slopes, slope_keys, mu_slope_forward, sigma_slope_forward
    )
    intercepts_inverse = jax.vmap(sample_latent_inverse, in_axes=(None, 0, 0, 0))(
        intercepts, intercept_keys, mu_intercept_forward, sigma_intercept_forward
    )

    return Parameters(
        mu_slope_inverse,
        mu_intercept_inverse,
        sigma_slope_inverse,
        sigma_intercept_inverse,
        sigma_obs_inverse,
        slopes_inverse,
        intercepts_inverse,
    )


def log_density_fn(
    position,
    mu_slope,
    mu_intercept,
    sigma_slope,
    sigma_intercept,
    sigma_obs,
    slopes,
    intercepts,
    X,
    y,
    group_idx,
):
    mu_slope_forward, mu_slope_log_prob = apply_prior(mu_slope, position.mu_slope)
    mu_intercept_forward, mu_intercept_log_prob = apply_prior(
        mu_intercept, position.mu_intercept
    )
    sigma_slope_forward, sigma_slope_log_prob = apply_prior(
        sigma_slope, position.sigma_slope
    )
    sigma_intercept_forward, sigma_intercept_log_prob = apply_prior(
        sigma_intercept, position.sigma_intercept
    )
    sigma_obs_forward, sigma_obs_log_prob = apply_prior(
        sigma_obs, position.sigma_obs
    )

    slopes_forward, slopes_log_prob = apply_latent(
        slopes, position.slopes, mu_slope_forward, sigma_slope_forward
    )
    intercepts_forward, intercepts_log_prob = apply_latent(
        intercepts, position.intercepts, mu_intercept_forward, sigma_intercept_forward
    )

    likelihood = HierarchicalLikelihood(X, group_idx)(
        slopes_forward, intercepts_forward, sigma_obs_forward
    )
    log_likelihood = likelihood.log_prob(y).sum()

    return (
        mu_slope_log_prob
        + mu_intercept_log_prob
        + sigma_slope_log_prob
        + sigma_intercept_log_prob
        + sigma_obs_log_prob
        + slopes_log_prob
        + intercepts_log_prob
        + log_likelihood
    )


def sample_prior_predictive(
    key,
    mu_slope,
    mu_intercept,
    sigma_slope,
    sigma_intercept,
    sigma_obs,
    slopes,
    intercepts,
):
    keys = jax.random.split(key, 7)
    mu_slope_forward = mu_slope.sample(keys[0])
    mu_intercept_forward = mu_intercept.sample(keys[1])
    sigma_slope_forward = sigma_slope.sample(keys[2])
    sigma_intercept_forward = sigma_intercept.sample(keys[3])
    sigma_obs_forward = sigma_obs.sample(keys[4])

    slopes_forward = slopes.sample(
        keys[5], 1, mu_slope_forward, sigma_slope_forward
    )
    intercepts_forward = intercepts.sample(
        keys[6], 1, mu_intercept_forward, sigma_intercept_forward
    )
    return slopes_forward, intercepts_forward, sigma_obs_forward


true_params = {
    "mu_slope": 0.8,
    "sigma_slope": 0.3,
    "mu_intercept": 0.2,
    "sigma_intercept": 0.2,
    "sigma_obs": 0.1
}

num_groups = 5
num_per_group = 20
key = jax.random.PRNGKey(42)
key, data_key = jax.random.split(key)
X, y, group_idx, true_slopes, true_intercepts = generate_hierarchical_data(
    data_key, num_groups, num_per_group, true_params
)

print("True parameters:")
print(f"  mu_slope: {true_params['mu_slope']}, sigma_slope: {true_params['sigma_slope']}")
print(f"  mu_intercept: {true_params['mu_intercept']}, sigma_intercept: {true_params['sigma_intercept']}")
print(f"  sigma_obs: {true_params['sigma_obs']}")
print(f"  true_slopes: {true_slopes}")
print(f"  true_intercepts: {true_intercepts}")

mu_slope = paz.Prior("mu_slope", tfd.Normal(0.0, 1.0))
mu_intercept = paz.Prior("mu_intercept", tfd.Normal(0.0, 1.0))

low, high = 0.01, 1.0
sigma_bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
sigma_slope = paz.Prior("sigma_slope", tfd.Uniform(low, high), bijector=sigma_bijector)
sigma_intercept = paz.Prior("sigma_intercept", tfd.Uniform(low, high), bijector=sigma_bijector)

slopes = paz.Latent("slopes", SlopePrior(num_groups))(mu_slope, sigma_slope)
intercepts = paz.Latent("intercepts", InterceptPrior(num_groups))(mu_intercept, sigma_intercept)

obs_low, obs_high = 0.01, 0.5
obs_bijector = tfb.Chain([tfb.Shift(obs_low), tfb.Scale(obs_high - obs_low), tfb.Sigmoid()])
sigma_obs = paz.Prior("sigma_obs", tfd.Uniform(obs_low, obs_high), bijector=obs_bijector)

print("\nSampling from prior predictive...")
key, prior_key = jax.random.split(key)
colors = plt.cm.tab10(jp.arange(num_groups))
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for key_i in jax.random.split(prior_key, 20):
    slopes_forward, intercepts_forward, _ = sample_prior_predictive(
        key_i,
        mu_slope,
        mu_intercept,
        sigma_slope,
        sigma_intercept,
        sigma_obs,
        slopes,
        intercepts,
    )
    for j in range(num_groups):
        mask = group_idx == j
        x_j = X[mask]
        sort_idx = jp.argsort(x_j)
        y_pred = slopes_forward[j] * x_j + intercepts_forward[j]
        plt.plot(x_j[sort_idx], y_pred[sort_idx], color=colors[j], alpha=0.1)
plt.title("Prior predictive")
plt.xlabel("X")
plt.ylabel("y")

plt.subplot(1, 2, 2)
for j in range(num_groups):
    mask = group_idx == j
    plt.scatter(X[mask], y[mask], color=colors[j], alpha=0.6, label=f"Group {j}")
plt.legend()
plt.title("Observed data")
plt.xlabel("X")
plt.ylabel("y")
plt.tight_layout()
plt.show()

num_chains = 4
num_samples = 3000
sigma = 0.02

key, init_key = jax.random.split(key)
positions = sample_initial_positions(
    init_key,
    num_chains,
    mu_slope,
    mu_intercept,
    sigma_slope,
    sigma_intercept,
    sigma_obs,
    slopes,
    intercepts,
)

print(f"\nRunning MCMC with {num_samples} samples, {num_chains} chains...")
key, mcmc_key = jax.random.split(key)
samples, infos = paz.metropolis_hastings.sample(
    mcmc_key,
    lambda params: log_density_fn(
        params,
        mu_slope,
        mu_intercept,
        sigma_slope,
        sigma_intercept,
        sigma_obs,
        slopes,
        intercepts,
        X,
        y,
        group_idx,
    ),
    positions,
    sigma,
    num_samples,
    num_chains,
)

burn_in = 1000
posterior = jax.tree.map(lambda x: x[burn_in:], samples)

print(f"\nMean acceptance rate: {infos.acceptance_rate[burn_in:].mean():.3f}")

print("\nPosterior estimates vs true values:")
print(f"  mu_slope: {posterior.position.mu_slope.mean():.3f} (true: {true_params['mu_slope']})")
print(f"  mu_intercept: {posterior.position.mu_intercept.mean():.3f} (true: {true_params['mu_intercept']})")
print(f"  sigma_slope: {sigma_bijector(posterior.position.sigma_slope).mean():.3f} (true: {true_params['sigma_slope']})")
print(f"  sigma_intercept: {sigma_bijector(posterior.position.sigma_intercept).mean():.3f} (true: {true_params['sigma_intercept']})")
print(f"  sigma_obs: {obs_bijector(posterior.position.sigma_obs).mean():.3f} (true: {true_params['sigma_obs']})")

posterior_slopes = posterior.position.slopes.reshape(-1, num_groups)
posterior_intercepts = posterior.position.intercepts.reshape(-1, num_groups)

print("\nGroup-level estimates:")
for j in range(num_groups):
    print(f"  Group {j}: slope={posterior_slopes[:, j].mean():.3f} (true: {true_slopes[j]:.3f}), "
          f"intercept={posterior_intercepts[:, j].mean():.3f} (true: {true_intercepts[j]:.3f})")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].hist(posterior.position.mu_slope.flatten(), bins=50, density=True, alpha=0.7)
axes[0, 0].axvline(true_params["mu_slope"], color="red", linestyle="--", label="True")
axes[0, 0].set_xlabel("mu_slope")
axes[0, 0].legend()

axes[0, 1].hist(posterior.position.mu_intercept.flatten(), bins=50, density=True, alpha=0.7)
axes[0, 1].axvline(true_params["mu_intercept"], color="red", linestyle="--", label="True")
axes[0, 1].set_xlabel("mu_intercept")
axes[0, 1].legend()

axes[0, 2].hist(sigma_bijector(posterior.position.sigma_slope).flatten(), bins=50, density=True, alpha=0.7)
axes[0, 2].axvline(true_params["sigma_slope"], color="red", linestyle="--", label="True")
axes[0, 2].set_xlabel("sigma_slope")
axes[0, 2].legend()

axes[1, 0].hist(sigma_bijector(posterior.position.sigma_intercept).flatten(), bins=50, density=True, alpha=0.7)
axes[1, 0].axvline(true_params["sigma_intercept"], color="red", linestyle="--", label="True")
axes[1, 0].set_xlabel("sigma_intercept")
axes[1, 0].legend()

axes[1, 1].hist(obs_bijector(posterior.position.sigma_obs).flatten(), bins=50, density=True, alpha=0.7)
axes[1, 1].axvline(true_params["sigma_obs"], color="red", linestyle="--", label="True")
axes[1, 1].set_xlabel("sigma_obs")
axes[1, 1].legend()

for j in range(num_groups):
    axes[1, 2].scatter(
        posterior_slopes[:, j].mean(),
        posterior_intercepts[:, j].mean(),
        color=colors[j], s=100, label=f"Group {j} (post)", zorder=3
    )
    axes[1, 2].scatter(
        true_slopes[j], true_intercepts[j],
        color=colors[j], marker="x", s=100, zorder=3
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
        idx = jax.random.randint(jax.random.PRNGKey(i), (), 0, len(posterior_slopes))
        slope_i = posterior_slopes[idx, j]
        intercept_i = posterior_intercepts[idx, j]
        plt.plot(x_plot, slope_i * x_plot + intercept_i, color=colors[j], alpha=0.1)

    plt.plot(x_plot, true_slopes[j] * x_plot + true_intercepts[j], "k--", linewidth=2, label="True")
    plt.title(f"Group {j}")
    plt.xlabel("X")
    if j == 0:
        plt.ylabel("y")
    plt.legend()

plt.suptitle("Posterior predictive by group", y=1.02)
plt.tight_layout()
plt.show()

print("\nShrinkage effect (group estimates pulled toward global mean):")
pooled_slope = posterior.position.mu_slope.mean()
pooled_intercept = posterior.position.mu_intercept.mean()
print(f"  Pooled slope mean: {pooled_slope:.3f}")
print(f"  Pooled intercept mean: {pooled_intercept:.3f}")
