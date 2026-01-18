import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.gaussian import build_gaussian_density
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


tfd = tfp.distributions


def _build_latent_space():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_gaussian_density_log_prob_shape():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    log_prob = density.log_prob(samples, space="inv")
    assert log_prob.shape == (5,)


def test_gaussian_density_sample_has_field():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    draw = density.sample(jax.random.PRNGKey(0), num_samples=1, space="inv")
    assert hasattr(draw, "x")
