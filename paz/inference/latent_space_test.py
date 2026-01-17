import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.prior import Prior
from paz.inference.latent_space import (
    build_latent_space,
    to_forward_samples,
    to_inverse_samples,
)

tfd = tfp.distributions
tfb = tfp.bijectors


def test_build_latent_space_preserves_names():
    mu = Prior(tfd.Normal(0.0, 1.0), name="mu")
    sigma = Prior(tfd.Normal(0.0, 1.0), name="sigma")
    latent_space = build_latent_space([mu, sigma])
    assert latent_space.names == ["mu", "sigma"]


def test_latent_space_round_trip():
    bijector = tfb.Exp()
    x = Prior(tfd.Normal(0.0, 1.0), name="x")
    y = Prior(tfd.Normal(0.0, 1.0), bijector=bijector, name="y")
    latent_space = build_latent_space([x, y])
    inverse = latent_space.Sample(x=jp.array(0.5), y=jp.array(-1.0))
    forward = to_forward_samples(latent_space, inverse)
    round_trip = to_inverse_samples(latent_space, forward)
    assert jp.allclose(round_trip.x, inverse.x)
    assert jp.allclose(round_trip.y, inverse.y)


def test_latent_space_accepts_dict_samples():
    x = Prior(tfd.Normal(0.0, 1.0), name="x")
    y = Prior(tfd.Normal(0.0, 1.0), name="y")
    latent_space = build_latent_space([x, y])
    forward = to_forward_samples(
        latent_space, {"x": jp.array(0.1), "y": jp.array(0.2)}
    )
    assert hasattr(forward, "x")
    assert hasattr(forward, "y")
