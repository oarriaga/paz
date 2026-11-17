# Fit Gaussian mixtures with EM: two cases the model fits well and a ring it
# cannot represent.

import argparse

import jax
import jax.numpy as jp

import paz.utils.plot as plot
from paz import distributions as pzd
from paz.inference.gmm.model import GMM


def sample_mixture_1D(key, num_samples):
    mixture = pzd.Categorical(probs=jp.array([0.7, 0.3]))
    components = pzd.Normal(jp.array([-2.0, 2.5]), jp.array([0.5, 0.4]))
    distribution = pzd.MixtureSameFamily(mixture, components)
    return distribution.sample(num_samples, seed=key)[:, None]


def sample_mixture_2D(key, num_samples):
    means = jp.array([[-3.0, -0.5], [3.0, 0.5]])
    covariance_0 = [[0.6, 0.2], [0.2, 0.4]]
    covariance_1 = [[0.5, -0.1], [-0.1, 0.3]]
    covariances = jp.array([covariance_0, covariance_1])
    mixture = pzd.Categorical(probs=jp.array([0.5, 0.5]))
    components = pzd.MultivariateNormalFullCovariance(means, covariances)
    distribution = pzd.MixtureSameFamily(mixture, components)
    return distribution.sample(num_samples, seed=key)


def sample_ring(key, num_samples):
    angle_key, radius_key = jax.random.split(key)
    shape = (num_samples,)
    angles = jax.random.uniform(angle_key, shape, maxval=2.0 * jp.pi)
    radius = 2.0 + 0.2 * jax.random.normal(radius_key, shape)
    return jp.stack([radius * jp.cos(angles), radius * jp.sin(angles)], axis=1)


def print_fit(name, mixture, samples):
    print(f"\n{name}")
    print("weights:", mixture.mixture_distribution.probs_parameter())
    print("means:", mixture.components_distribution.loc)
    print("avg_log_prob:", float(mixture.log_prob(samples).mean()))


def plot_fit_1D(axis, samples, mixture, title):
    weights = mixture.mixture_distribution.probs_parameter()
    components = mixture.components_distribution
    plot.gmm_1d(samples, weights, components.loc, components.stddev(), axis)
    axis.set_title(title)
    plot.set_labels(axis, x="x", y="density")
    plot.clean(axis)


def plot_fit_2D(axis, samples, mixture, title):
    weights = mixture.mixture_distribution.probs_parameter()
    components = mixture.components_distribution
    parameters = (weights, components.loc, components.covariance())
    plot.gmm_2d(samples, *parameters, axis, plot.BLUE_GREY.primary)
    axis.set_title(title)
    plot.set_labels(axis, x="x", y="y")
    plot.clean(axis, spines="box")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian mixture fits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iters", type=int, default=60)
    parser.add_argument("--num_ring_components", type=int, default=6)
    args = parser.parse_args()

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 6)
    kwargs = {"method": "em", "num_iters": args.num_iters}
    samples_1D = sample_mixture_1D(keys[0], 500)
    mixture_1D = GMM(2, covariance="diag").fit(keys[1], samples_1D, **kwargs)
    samples_2D = sample_mixture_2D(keys[2], 800)
    mixture_2D = GMM(2, covariance="full").fit(keys[3], samples_2D, **kwargs)
    ring = sample_ring(keys[4], 800)
    ring_model = GMM(args.num_ring_components, covariance="full")
    ring_mixture = ring_model.fit(keys[5], ring, **kwargs)
    print_fit("1D mixture (good fit)", mixture_1D, samples_1D)
    print_fit("2D mixture (good fit)", mixture_2D, samples_2D)
    print_fit("Ring data (limitations)", ring_mixture, ring)

    plot.configure(fontsize=12)
    figure, axes = plot.subplots(ncols=3, figsize=(15, 4))
    plot_fit_1D(axes[0], samples_1D, mixture_1D, "Good fit (1D)")
    plot_fit_2D(axes[1], samples_2D, mixture_2D, "Good fit (2D)")
    plot_fit_2D(axes[2], ring, ring_mixture, "Bad fit (ring)")
    figure.tight_layout()
    plot.show()
