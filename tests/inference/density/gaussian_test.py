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


def test_gaussian_density_samples_follow_the_fitted_moments():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    draws = density.sample_inverse(jax.random.PRNGKey(0), num_samples=4000)
    assert draws.x.shape == (4000,)
    assert jp.isclose(draws.x.mean(), 0.0, atol=0.05)
    assert jp.isclose(draws.x.std(), samples.x.std(), atol=0.05)


def test_gaussian_density_rejects_many_samples():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    with pytest.raises(ValueError, match="jax.vmap"):
        density.log_prob_inverse(samples)
    with pytest.raises(ValueError, match="jax.vmap"):
        density.prob(samples)
    assert density.log_prob_inverse({"x": jp.array(0.2)}).shape == ()


def test_gaussian_density_integrates_to_one_in_forward_space():
    lognormal = paz.distributions.LogNormal(0.0, 0.5)
    prior = Prior(lognormal, paz.bijectors.Exp(), "scale")
    latent_space = build_latent_space([prior])
    draws = 0.4 * jax.random.normal(jax.random.PRNGKey(0), (500,))
    density = build_gaussian_density(latent_space.Sample(draws), latent_space)
    grid = jp.linspace(0.01, 12.0, 6000)
    probs = jax.vmap(density.prob)({"scale": grid})
    assert jp.isclose(jp.trapezoid(probs, grid), 1.0, atol=1e-2)


def test_gaussian_density_forward_matches_inverse_without_bijector():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    density = build_gaussian_density(samples, latent_space)
    sample = {"x": jp.array(0.3)}
    expected = density.log_prob_inverse(sample)
    assert jp.isclose(density.log_prob(sample), expected)


def test_gaussian_density_forward_log_prob_of_vector_and_scalar_latents():
    normal = paz.distributions.Normal(0.0, 1.0)
    lognormal = paz.distributions.LogNormal(jp.zeros(2), 0.5)
    scales = Prior(lognormal, paz.bijectors.Exp(), "scales")
    latent_space = build_latent_space([Prior(normal, name="x"), scales])
    keys = jax.random.split(jax.random.PRNGKey(0))
    x_draws = jax.random.normal(keys[0], (200,))
    scale_draws = jax.random.normal(keys[1], (200, 2))
    draws = latent_space.Sample(x_draws, scale_draws)
    density = build_gaussian_density(draws, latent_space)
    forward = latent_space.Sample(x=jp.array(0.3), scales=jp.array([0.5, 2.0]))
    inverse = latent_space.Sample(x=forward.x, scales=jp.log(forward.scales))
    expected = density.log_prob_inverse(inverse) - jp.log(forward.scales).sum()
    assert density.log_prob(forward).shape == ()
    assert jp.isclose(density.log_prob(forward), expected)
