import jax.numpy as jp

import paz
from paz.inference.density import core
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


def _build_latent_space():
    prior = Prior(paz.distributions.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_flatten_samples_stacks_one_row_per_sample():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.array([0.0, 1.0, 2.0]))
    flat, _ = core._flatten_samples(samples, latent_space)
    assert jp.array_equal(flat, jp.array([[0.0], [1.0], [2.0]]))


def test_unflatten_samples_single():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.array([0.0, 1.0]))
    _, unravel = core._flatten_samples(samples, latent_space)
    result = core._unflatten_samples(unravel, jp.array([3.0]))
    assert result.x == 3.0


def test_compute_covariance_regularizes_the_sample_covariance():
    flat_samples = jp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    covariance = core._compute_covariance(flat_samples, 0.5)
    assert jp.allclose(covariance, jp.array([[1.5, 1.0], [1.0, 1.5]]))


def test_lowrank_covariance_keeps_the_largest_eigenvalues():
    covariance = jp.diag(jp.array([1.0, 2.0, 3.0]))
    lowrank = core._lowrank_covariance(covariance, 2, 0.5)
    expected = jp.diag(jp.array([0.5, 2.5, 3.5]))
    assert jp.allclose(lowrank, expected, atol=1e-5)
