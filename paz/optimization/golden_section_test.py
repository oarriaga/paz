import jax
import jax.numpy as jp

from paz.optimization import golden_section_search

NUM_STEPS = 20


def compute_bound(low, high, num_steps=NUM_STEPS):
    return (0.618**num_steps) * (high - low)


def test_golden_section_search_finds_quadratic_minimum():
    def function(x):
        return (x - 0.3) ** 2

    low, high = jp.array(-2.0), jp.array(2.0)
    x = golden_section_search(function, low, high, NUM_STEPS)
    assert abs(float(x) - 0.3) < compute_bound(-2.0, 2.0)


def test_golden_section_search_finds_minimum_at_low_edge():
    low, high = jp.array(0.0), jp.array(1.0)
    x = golden_section_search(jp.exp, low, high, NUM_STEPS)
    assert abs(float(x)) < compute_bound(0.0, 1.0)


def test_golden_section_search_finds_non_quadratic_minimum():
    def function(x):
        return jp.exp(x) - (2.0 * x)

    low, high = jp.array(-1.0), jp.array(2.0)
    x = golden_section_search(function, low, high, NUM_STEPS)
    assert abs(float(x) - float(jp.log(2.0))) < compute_bound(-1.0, 2.0)


def test_golden_section_search_works_under_jit():
    @jax.jit
    def search(center):
        def function(x):
            return (x - center) ** 2

        low, high = jp.array(-2.0), jp.array(2.0)
        return golden_section_search(function, low, high, NUM_STEPS)

    x = search(jp.array(-0.7))
    assert abs(float(x) + 0.7) < compute_bound(-2.0, 2.0)


def test_golden_section_search_works_inside_fori_loop():
    centers = jp.array([-0.5, 0.25, 1.5])

    def search(index, minima):
        def function(x):
            return (x - centers[index]) ** 2

        low, high = jp.array(-2.0), jp.array(2.0)
        x = golden_section_search(function, low, high, NUM_STEPS)
        return minima.at[index].set(x)

    minima = jax.lax.fori_loop(0, 3, search, jp.zeros(3))
    assert jp.all(jp.abs(minima - centers) < compute_bound(-2.0, 2.0))


def test_golden_section_search_returns_float32_scalar():
    def function(x):
        return (x - 0.3) ** 2

    low, high = jp.array(-2.0), jp.array(2.0)
    x = golden_section_search(function, low, high, NUM_STEPS)
    assert x.shape == ()
    assert x.dtype == jp.float32
