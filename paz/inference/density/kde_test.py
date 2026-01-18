import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.kde import build_kde_density
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


tfd = tfp.distributions


def _build_latent_space():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_kde_density_log_prob_shape():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 6))
    density = build_kde_density(samples, latent_space)
    log_prob = density.log_prob(samples, space="inv")
    assert log_prob.shape == (6,)


def test_kde_density_sample_shape():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 6))
    density = build_kde_density(samples, latent_space)
    draw = density.sample(jax.random.PRNGKey(1), num_samples=2, space="inv")
    assert draw.x.shape == (2,)
