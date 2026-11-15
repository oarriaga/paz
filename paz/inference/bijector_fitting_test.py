import jax
import jax.numpy as jp

from paz.backend import distributions
from paz.backend import bijectors
from paz.inference.bijector_fitting import fit_bijector


def test_fit_bijector_reduces_loss():
    key = jax.random.PRNGKey(0)
    source = distributions.Normal(0.0, 1.0)
    target = distributions.Normal(2.0, 0.5)
    initial = bijectors.Chain([bijectors.Shift(0.0), bijectors.Scale(1.0)])
    _, losses = fit_bijector(
        source,
        target,
        initial,
        key,
        num_samples=256,
        num_steps=40,
        verbose=False,
    )
    assert losses[-1] <= losses[0]


def test_fit_bijector_returns_bijector():
    key = jax.random.PRNGKey(1)
    source = distributions.Normal(0.0, 1.0)
    target = distributions.Normal(-1.0, 2.0)
    initial = bijectors.Chain([bijectors.Shift(0.0), bijectors.Scale(1.0)])
    fitted, _ = fit_bijector(
        source,
        target,
        initial,
        key,
        num_samples=128,
        num_steps=5,
        verbose=False,
    )
    assert isinstance(fitted, bijectors.Bijector)
