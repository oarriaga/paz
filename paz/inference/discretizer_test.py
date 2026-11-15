import jax
import jax.numpy as jp

from paz.backend import distributions
from paz.inference.discretizer import (
    discretize,
    get_grid_values,
    indices_to_values,
)


def test_get_grid_values_endpoints():
    grid = get_grid_values(-2.0, 3.0, 6)
    assert jp.allclose(grid[0], -2.0)
    assert jp.allclose(grid[-1], 3.0)


def test_indices_to_values_matches_grid():
    min_val, max_val, num_steps = -1.5, 2.5, 9
    indices = jp.arange(num_steps)
    values = indices_to_values(indices, min_val, max_val, num_steps)
    grid = get_grid_values(min_val, max_val, num_steps)
    assert jp.allclose(values, grid)


def test_discretize_probabilities_sum_to_one():
    distribution = distributions.Normal(loc=0.0, scale=1.0)
    categorical = discretize(distribution, -3.0, 3.0, 31)
    probs = categorical.probs_parameter()
    assert probs.shape == (31,)
    assert jp.all(jp.isfinite(probs))
    assert jp.allclose(jp.sum(probs), 1.0)


def test_discretize_symmetry_for_standard_normal():
    distribution = distributions.Normal(loc=0.0, scale=1.0)
    categorical = discretize(distribution, -2.5, 2.5, 21)
    probs = categorical.probs_parameter()
    assert jp.allclose(probs, probs[::-1], atol=1e-4)


def test_discretize_peak_at_center():
    distribution = distributions.Normal(loc=0.0, scale=1.0)
    categorical = discretize(distribution, -3.0, 3.0, 31)
    probs = categorical.probs_parameter()
    assert int(jp.argmax(probs)) == 15


def test_indices_to_values_accepts_samples():
    key = jax.random.PRNGKey(0)
    distribution = distributions.Normal(loc=0.0, scale=1.0)
    categorical = discretize(distribution, -1.0, 1.0, 11)
    indices = categorical.sample(5, seed=key)
    values = indices_to_values(indices, -1.0, 1.0, 11)
    assert values.shape == (5,)


def test_discretize_log_normal_has_finite_probs():
    distribution = distributions.LogNormal(loc=0.0, scale=0.6)
    categorical = discretize(distribution, 0.0, 6.0, 25)
    probs = categorical.probs_parameter()
    assert jp.all(jp.isfinite(probs))
