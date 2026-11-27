import jax
import jax.numpy as jp

import paz
from paz.inference.density.kde import build_kde_density
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


def _build_latent_space():
    prior = Prior(paz.distributions.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_kde_density_integrates_to_one():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 6))
    density = build_kde_density(samples, latent_space)
    grid = jp.linspace(-6.0, 6.0, 4000)
    probs = jax.vmap(density.prob)({"x": grid})
    assert jp.isclose(jp.trapezoid(probs, grid), 1.0, atol=1e-2)


def test_kde_density_sample_shape():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 6))
    density = build_kde_density(samples, latent_space)
    draw = density.sample_inverse(jax.random.PRNGKey(1), num_samples=2)
    assert draw.x.shape == (2,)


def test_kde_density_scores_one_vector_sample():
    normal = paz.distributions.Normal(jp.zeros(2), 1.0)
    latent_space = build_latent_space([Prior(normal, name="x")])
    draws = jax.random.normal(jax.random.PRNGKey(0), (30, 2))
    samples = latent_space.Sample(x=draws)
    density = build_kde_density(samples, latent_space)
    assert density.log_prob_inverse({"x": jp.zeros(2)}).shape == ()
    assert jax.vmap(density.log_prob)(samples).shape == (30,)
