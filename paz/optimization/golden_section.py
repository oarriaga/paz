import jax
import jax.numpy as jp
import numpy as np


def golden_section_search(function, low, high, num_steps):
    """Minimizes a unimodal one-dimensional function on [low, high].

    Each step shrinks the bracket by the golden ratio (~0.618), so after
    `num_steps` the returned point is within 0.618**num_steps * (high - low)
    of the minimum. Runs in `jax.lax.fori_loop` and is jit-compatible.
    """
    ratio = 0.5 * (np.sqrt(5.0) - 1.0)

    def narrow(_, bracket):
        low, high = bracket
        left = high - (ratio * (high - low))
        right = low + (ratio * (high - low))
        is_left = function(left) < function(right)
        return jp.where(is_left, low, left), jp.where(is_left, right, high)

    low, high = jax.lax.fori_loop(0, num_steps, narrow, (low, high))
    return 0.5 * (low + high)
