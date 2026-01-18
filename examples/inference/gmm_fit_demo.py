import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

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


def _build_initial_parameters(num_components, num_dims):
    weights = jp.full((num_components,), 1.0 / num_components)
    means = jp.zeros((num_components, num_dims))
    covariances = jp.ones((num_components, num_dims))
    return weights, means, covariances


def _extract_parameters(model, key):
    samples = model.sample_inverse(key)
    return samples.weights, samples.means, samples.covariances


def _plot_1d_fit(axis, samples, weights, means, covariances, covariance, title):
    samples = jp.squeeze(samples)
    axis.hist(samples, bins=40, density=True, color="0.7", alpha=0.6)
    x_min = float(samples.min() - 1.0)
    x_max = float(samples.max() + 1.0)
    xs = jp.linspace(x_min, x_max, 400)
    distribution = _build_gmm_distribution(
        weights, means, covariances, covariance
    )
    density = jp.exp(distribution.log_prob(xs[:, None]))
    axis.plot(xs, density, color="black", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("density")


def _plot_2d_fit(axis, samples, weights, means, covariances, covariance, title):
    samples = jp.asarray(samples)
    axis.scatter(
        samples[:, 0],
        samples[:, 1],
        s=8,
        alpha=0.35,
        color="tab:blue",
    )
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
    density = jp.exp(distribution.log_prob(grid)).reshape(grid_x.shape)
    axis.contour(grid_x, grid_y, density, levels=10, colors="black")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")


def _describe_fit(name, samples, weights, means, covariances, covariance):
    distribution = _build_gmm_distribution(
        weights, means, covariances, covariance
    )
    avg_log_prob = jp.mean(distribution.log_prob(samples))
    print(f"\n{name}")
    print("weights:", weights)
    print("means:", means)
    print("avg_log_prob:", float(avg_log_prob))


def main():
    key = jax.random.PRNGKey(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    key, sample_key = jax.random.split(key)
    samples_1d = _sample_1d_mixture(sample_key, 500)
    weights, means, covariances = _build_initial_parameters(2, 1)
    model = GMM(weights, means, covariances, covariance="diag", name="gmm_1d")
    key, fit_key, read_key = jax.random.split(key, 3)
    fitted = model.fit(fit_key, samples_1d, method="em", num_iters=60)
    weights, means, covariances = _extract_parameters(fitted, read_key)
    _describe_fit(
        "1D mixture (good fit)", samples_1d, weights, means, covariances, "diag"
    )
    _plot_1d_fit(
        axes[0],
        samples_1d,
        weights,
        means,
        covariances,
        "diag",
        "Good fit (1D)",
    )

    key, sample_key = jax.random.split(key)
    samples_2d = _sample_2d_mixture(sample_key, 800)
    weights, means, covariances = _build_initial_parameters(2, 2)
    model = GMM(weights, means, covariances, covariance="full", name="gmm_2d")
    key, fit_key, read_key = jax.random.split(key, 3)
    fitted = model.fit(fit_key, samples_2d, method="em", num_iters=60)
    weights, means, covariances = _extract_parameters(fitted, read_key)
    _describe_fit(
        "2D mixture (good fit)", samples_2d, weights, means, covariances, "full"
    )
    _plot_2d_fit(
        axes[1],
        samples_2d,
        weights,
        means,
        covariances,
        "full",
        "Good fit (2D)",
    )

    key, sample_key = jax.random.split(key)
    samples_ring = _sample_ring(sample_key, 800)
    weights, means, covariances = _build_initial_parameters(6, 2)
    model = GMM(weights, means, covariances, covariance="full", name="gmm_ring")
    key, fit_key, read_key = jax.random.split(key, 3)
    fitted = model.fit(fit_key, samples_ring, method="em", num_iters=60)
    weights, means, covariances = _extract_parameters(fitted, read_key)
    _describe_fit(
        "Ring data (limitations)",
        samples_ring,
        weights,
        means,
        covariances,
        "full",
    )
    _plot_2d_fit(
        axes[2],
        samples_ring,
        weights,
        means,
        covariances,
        "full",
        "Bad fit (ring)",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
