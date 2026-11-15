import jax
import jax.numpy as jp

import paz
from paz.inference.bijector_fitting import fit_bijector


def test_fit_bijector_reduces_loss():
    key = jax.random.PRNGKey(0)
    source = paz.distributions.Normal(0.0, 1.0)
    target = paz.distributions.Normal(2.0, 0.5)
    affine = [paz.bijectors.Shift(0.0), paz.bijectors.Scale(1.0)]
    initial = paz.bijectors.Chain(affine)
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
    source = paz.distributions.Normal(0.0, 1.0)
    target = paz.distributions.Normal(-1.0, 2.0)
    affine = [paz.bijectors.Shift(0.0), paz.bijectors.Scale(1.0)]
    initial = paz.bijectors.Chain(affine)
    fitted, _ = fit_bijector(
        source,
        target,
        initial,
        key,
        num_samples=128,
        num_steps=5,
        verbose=False,
    )
    assert isinstance(fitted, paz.bijectors.Bijector)
