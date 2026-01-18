import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.density import core
from paz.inference.latent_space import build_latent_space
from paz.inference.prior import Prior


tfd = tfp.distributions


def _build_latent_space():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    return build_latent_space([prior])


def test_flatten_samples_batch_size():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.array([0.0, 1.0, 2.0]))
    flat, _ = core._flatten_samples(samples, latent_space)
    assert flat.shape[0] == 3


def test_unflatten_samples_single():
    latent_space = _build_latent_space()
    samples = latent_space.Sample(x=jp.array([0.0, 1.0]))
    _, unravel = core._flatten_samples(samples, latent_space)
    result = core._unflatten_samples(unravel, jp.array([3.0]))
    assert result.x == 3.0


def test_compute_covariance_shape():
    flat_samples = jp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    cov = core._compute_covariance(flat_samples, 1e-6)
    assert cov.shape == (2, 2)


def test_lowrank_covariance_shape():
    cov = jp.eye(3)
    lowrank = core._lowrank_covariance(cov, rank=2, regularization=1e-6)
    assert lowrank.shape == (3, 3)
