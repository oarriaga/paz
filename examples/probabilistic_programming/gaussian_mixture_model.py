import jax
import jax.numpy as jp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import paz.plot as plot
from paz.inference.gmm.model import GMM


tfd = tfp.distributions


def _build_gmm_distribution(weights, means, covariances, covariance):
    if covariance == "diag":
        components = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jp.sqrt(covariances)
        )
    else:
        components = tfd.MultivariateNormalFullCovariance(
            loc=means, covariance_matrix=covariances
        )
    return tfd.MixtureSameFamily(tfd.Categorical(probs=weights), components)


def _sample_1d_mixture(key, num_samples):
    weights = jp.array([0.7, 0.3])
    means = jp.array([-2.0, 2.5])
    stdvs = jp.array([0.5, 0.4])
    distribution = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights),
        tfd.Normal(loc=means, scale=stdvs),
    )
    return distribution.sample(num_samples, seed=key)[:, None]


def _sample_2d_mixture(key, num_samples):
    weights = jp.array([0.5, 0.5])
    means = jp.array([[-3.0, -0.5], [3.0, 0.5]])
    covariances = jp.array(
        [
            [[0.6, 0.2], [0.2, 0.4]],
            [[0.5, -0.1], [-0.1, 0.3]],
        ]
    )
    distribution = tfd.MixtureSameFamily(
        tfd.Categorical(probs=weights),
        tfd.MultivariateNormalFullCovariance(
            loc=means, covariance_matrix=covariances
        ),
    )
    return distribution.sample(num_samples, seed=key)


def _sample_ring(key, num_samples):
    key, angle_key = jax.random.split(key)
    angles = jax.random.uniform(
        angle_key, (num_samples,), minval=0.0, maxval=2.0 * jp.pi
    )
    key, radius_key = jax.random.split(key)
    radius = 2.0 + 0.2 * jax.random.normal(radius_key, (num_samples,))
    x = radius * jp.cos(angles)
    y = radius * jp.sin(angles)
    return jp.stack([x, y], axis=1)


def _get_mixture_probs(mixture_distribution):
    if hasattr(mixture_distribution, "probs_parameter"):
        return mixture_distribution.probs_parameter()
    if hasattr(mixture_distribution, "probs"):
        return mixture_distribution.probs
    return jax.nn.softmax(mixture_distribution.logits)


def _extract_parameters(distribution):
    weights = _get_mixture_probs(distribution.mixture_distribution)
    components = distribution.components_distribution
    means = components.loc
    if isinstance(components, tfd.MultivariateNormalDiag):
        covariances = components.variance()
    else:
        covariances = components.covariance()
    return weights, means, covariances


def _plot_1d_fit(axis, samples, weights, means, covariances, covariance, title):
    """Plot 1D GMM fit: histogram + density."""
    samples = jp.squeeze(samples)
    samples_np = np.array(samples)

    plot.histogram(
        samples_np, axis, bins=40, color=plot.BLUE_GREY.neutral, alpha=0.6
    )

    x_min = float(samples.min() - 1.0)
    x_max = float(samples.max() + 1.0)
    xs = jp.linspace(x_min, x_max, 400)
    distribution = _build_gmm_distribution(
        weights, means, covariances, covariance
    )
    density = np.array(jp.exp(distribution.log_prob(xs[:, None])))
    plot.line(np.array(xs), density, axis, color="black", linewidth=2)
    axis.set_title(title)
    plot.set_labels(axis, x="x", y="density")
    plot.clean(axis)


def _plot_2d_fit(axis, samples, weights, means, covariances, covariance, title):
    """Plot 2D GMM fit: scatter + contours."""
    samples = jp.asarray(samples)
    samples_np = np.array(samples)

    plot.scatter(samples_np[:, 0], samples_np[:, 1], axis, s=8, alpha=0.35,
                 color=plot.BLUE_GREY.primary)

    x_min = float(samples[:, 0].min() - 1.0)
    x_max = float(samples[:, 0].max() + 1.0)
    y_min = float(samples[:, 1].min() - 1.0)
    y_max = float(samples[:, 1].max() + 1.0)
    xs = jp.linspace(x_min, x_max, 120)
    ys = jp.linspace(y_min, y_max, 120)
    grid_x, grid_y = jp.meshgrid(xs, ys)
    grid = jp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    distribution = _build_gmm_distribution(
        weights, means, covariances, covariance
    )
    density = np.array(
        jp.exp(distribution.log_prob(grid)).reshape(grid_x.shape)
    )
    axis.contour(
        np.array(grid_x), np.array(grid_y), density, levels=10, colors="black"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    plot.set_labels(axis, x="x", y="y")
    plot.clean(axis, spines="box")


def _describe_fit(name, samples, weights, means, covariances, covariance):
    distribution = _build_gmm_distribution(
        weights, means, covariances, covariance
    )
    avg_log_prob = jp.mean(distribution.log_prob(samples))
    print(f"\n{name}")
    print("weights:", weights)
    print("means:", means)
    print("avg_log_prob:", float(avg_log_prob))


# Configure plotting
plot.configure(fontsize=12, latex=False)

key = jax.random.PRNGKey(0)

# 1D mixture fit
key, sample_key = jax.random.split(key)
samples_1d = _sample_1d_mixture(sample_key, 500)
model = GMM(2, covariance="diag", name="gmm_1d")
key, fit_key = jax.random.split(key)
fitted = model.fit(fit_key, samples_1d, method="em", num_iters=60)
weights_1d, means_1d, covariances_1d = _extract_parameters(fitted)
_describe_fit(
    "1D mixture (good fit)",
    samples_1d,
    weights_1d,
    means_1d,
    covariances_1d,
    "diag",
)

# 2D mixture fit
key, sample_key = jax.random.split(key)
samples_2d = _sample_2d_mixture(sample_key, 800)
model = GMM(2, covariance="full", name="gmm_2d")
key, fit_key = jax.random.split(key)
fitted = model.fit(fit_key, samples_2d, method="em", num_iters=60)
weights_2d, means_2d, covariances_2d = _extract_parameters(fitted)
_describe_fit(
    "2D mixture (good fit)",
    samples_2d,
    weights_2d,
    means_2d,
    covariances_2d,
    "full",
)

# Ring data (showing GMM limitations)
key, sample_key = jax.random.split(key)
samples_ring = _sample_ring(sample_key, 800)
model = GMM(6, covariance="full", name="gmm_ring")
key, fit_key = jax.random.split(key)
fitted = model.fit(fit_key, samples_ring, method="em", num_iters=60)
weights_ring, means_ring, covariances_ring = _extract_parameters(fitted)
_describe_fit(
    "Ring data (limitations)",
    samples_ring,
    weights_ring,
    means_ring,
    covariances_ring,
    "full",
)

# Create side-by-side figure showing all three fits
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

_plot_1d_fit(
    axes[0], samples_1d, weights_1d, means_1d, covariances_1d, "diag",
    "Good fit (1D)"
)
_plot_2d_fit(
    axes[1], samples_2d, weights_2d, means_2d, covariances_2d, "full",
    "Good fit (2D)"
)
_plot_2d_fit(
    axes[2], samples_ring, weights_ring, means_ring, covariances_ring, "full",
    "Bad fit (ring)"
)

plt.tight_layout()
plot.show()

# Individual larger plots for better visualization
fig, ax = plot.subplots(figsize=(8, 5))
_plot_1d_fit(ax, samples_1d, weights_1d, means_1d, covariances_1d, "diag",
             "1D Gaussian Mixture Model")
plot.show()

fig, ax = plot.subplots(figsize=(8, 6))
_plot_2d_fit(ax, samples_2d, weights_2d, means_2d, covariances_2d, "full",
             "2D Gaussian Mixture Model")
plot.show()

fig, ax = plot.subplots(figsize=(8, 6))
_plot_2d_fit(
    ax,
    samples_ring,
    weights_ring,
    means_ring,
    covariances_ring,
    "full",
    "GMM fit on ring data (showing limitations)",
)
plot.show()
