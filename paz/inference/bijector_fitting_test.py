import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.bijector_fitting import fit_bijector


tfd = tfp.distributions
tfb = tfp.bijectors


def test_fit_bijector_reduces_loss():
    key = jax.random.PRNGKey(0)
    source = tfd.Normal(0.0, 1.0)
    target = tfd.Normal(2.0, 0.5)
    initial = tfb.Chain([tfb.Shift(0.0), tfb.Scale(1.0)])
    _, losses = fit_bijector(
        source,
        target,
        initial,
        key,
        num_samples=256,
        num_steps=40,
        print=False,
    )
    assert losses[-1] <= losses[0]


def test_fit_bijector_returns_bijector():
    key = jax.random.PRNGKey(1)
    source = tfd.Normal(0.0, 1.0)
    target = tfd.Normal(-1.0, 2.0)
    initial = tfb.Chain([tfb.Shift(0.0), tfb.Scale(1.0)])
    fitted, _ = fit_bijector(
        source,
        target,
        initial,
        key,
        num_samples=128,
        num_steps=5,
        print=False,
    )
    assert isinstance(fitted, tfb.Bijector)
