from collections import namedtuple

import jax
import jax.numpy as jp

from .bijectors import (
    Chain,
    Identity,
    Scale,
    Shift,
    Sigmoid,
)
from .distributions import (
    Independent,
    Normal,
    TransformedDistribution,
    Uniform,
)
from .fitting import fit_bijector
from .utils import LOG_TWO_PI, log_prob_inverse


def test_normal_log_prob_matches_formula():
    distribution = Normal(0.5, 2.0)
    values = jp.array([-1.0, 0.5, 3.0])
    diff = (values - 0.5) / 2.0
    expected = -0.5 * LOG_TWO_PI - jp.log(2.0) - 0.5 * diff**2
    assert jp.allclose(distribution.log_prob(values), expected)


def test_uniform_log_prob_masks_values_outside_support():
    distribution = Uniform(-1.0, 1.0)
    values = jp.array([-2.0, 0.0, 3.0])
    log_prob = distribution.log_prob(values)
    assert jp.isneginf(log_prob[0])
    assert jp.isfinite(log_prob[1])
    assert jp.isneginf(log_prob[2])


def test_chain_round_trip_matches_sigmoid_bounds():
    bijector = Chain([Shift(-2.0), Scale(4.0), Sigmoid()])
    inverse = jp.array([-3.0, 0.0, 2.0])
    forward = bijector(inverse)
    recovered = bijector.inverse(forward)
    assert jp.allclose(recovered, inverse, atol=1e-5)


def test_identity_and_distribution_are_pytrees():
    Sample = namedtuple("Sample", ["distribution", "bijector"])
    pytree = Sample(Normal(jp.array([0.0, 1.0]), 1.0), Identity())
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt.distribution, Normal)
    assert isinstance(rebuilt.bijector, Identity)


def test_independent_sums_last_batch_dimension():
    base = Normal(jp.array([0.0, 1.0]), 1.0)
    distribution = Independent(base, 1)
    values = jp.array([0.0, 1.0])
    expected = base.log_prob(values).sum()
    assert jp.allclose(distribution.log_prob(values), expected)


def test_transformed_distribution_matches_change_of_variables():
    base = Normal(0.0, 1.0)
    bijector = Chain([Shift(1.5), Scale(0.5)])
    distribution = TransformedDistribution(base, bijector)
    values = jp.array([0.5, 1.0, 1.5])
    inverse = bijector.inverse(values)
    expected = base.log_prob(inverse) - jp.log(0.5)
    assert jp.allclose(distribution.log_prob(values), expected)


def test_log_prob_inverse_adds_log_det_jacobian():
    distribution = Uniform(0.1, 0.9)
    bijector = Chain([Shift(0.1), Scale(0.8), Sigmoid()])
    inverse = jp.array([-1.0, 0.0, 1.0])
    forward = bijector(inverse)
    expected = distribution.log_prob(forward)
    expected = expected + bijector.forward_log_det_jacobian(inverse)
    actual = log_prob_inverse(distribution, bijector, inverse)
    assert jp.allclose(actual, expected)


def test_fit_bijector_recovers_affine_target():
    key = jax.random.PRNGKey(13)
    source = Normal(0.0, 1.0)
    target = Normal(2.0, 0.5)
    initial = Chain([Shift(0.0), Scale(1.0)])
    fitted, losses = fit_bijector(
        source, target, initial, key, num_samples=4000, num_steps=1200
    )
    assert losses[-1] < losses[0]
    samples = fitted(source.sample(2048, seed=jax.random.PRNGKey(7)))
    assert abs(float(samples.mean()) - 2.0) < 0.15
    assert abs(float(samples.std()) - 0.5) < 0.15
