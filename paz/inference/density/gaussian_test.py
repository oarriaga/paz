import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.density.gaussian import build_gaussian_density
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


def _build_latent_space():
    prior = Prior(paz.distributions.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_gaussian_density_log_prob_shape():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    log_prob = jax.vmap(density.log_prob_inverse)(samples)
    assert log_prob.shape == (5,)


def test_gaussian_density_sample_has_field():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    draw = density.sample_inverse(jax.random.PRNGKey(0), num_samples=1)
    assert hasattr(draw, "x")


def test_gaussian_density_rejects_many_samples():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    with pytest.raises(ValueError, match="jax.vmap"):
        density.log_prob_inverse(samples)
    with pytest.raises(ValueError, match="jax.vmap"):
        density.prob(samples)
    assert density.log_prob_inverse({"x": jp.array(0.2)}).shape == ()
