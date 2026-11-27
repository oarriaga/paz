import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.density.gmm import build_gmm_density
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


def _build_latent_space():
    prior = Prior(paz.distributions.Normal(0.0, 1.0), name="x")
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
    log_prob = jax.vmap(density.log_prob_inverse)(samples)
    assert log_prob.shape == (6,)
