import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density.gmm import build_gmm_density
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


tfd = tfp.distributions


def _build_latent_space():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_gmm_density_requires_key():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 6))
    with pytest.raises(ValueError):
        build_gmm_density(None, samples, latent_space, k=2)


def test_gmm_density_log_prob_shape():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 6))
    density = build_gmm_density(jax.random.PRNGKey(0), samples, latent_space, k=2, num_iters=2)
    log_prob = density.log_prob_inverse(samples)
    assert log_prob.shape == (6,)
