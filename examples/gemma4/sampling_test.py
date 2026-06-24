import jax
import jax.numpy as jnp
import numpy as np

from .sampling import (SamplingArgs, apply_top_k, apply_top_p, sample_logits)


def test_top_k_masks_below_threshold():
    logits = jnp.asarray([[5.0, 4.0, 3.0, 2.0, 1.0]])
    out = np.asarray(apply_top_k(logits, 2))
    assert np.isinf(out[0, 2:]).all() and out[0, 2] < 0
    assert np.array_equal(out[0, :2], [5.0, 4.0])


def test_top_p_keeps_smallest_set_over_p():
    # softmax([3,2,1,0]) ~ [0.643, 0.236, 0.087, 0.032]; p=0.8 keeps first two.
    logits = jnp.asarray([[3.0, 2.0, 1.0, 0.0]])
    out = np.asarray(apply_top_p(logits, 0.8))
    assert np.isfinite(out[0, :2]).all()
    assert np.isinf(out[0, 2:]).all()


def test_top_p_always_keeps_top_token():
    logits = jnp.asarray([[10.0, 0.0, 0.0]])
    out = np.asarray(apply_top_p(logits, 0.1))
    assert np.isfinite(out[0, 0]) and np.isinf(out[0, 1:]).all()


def test_sample_shapes_and_range():
    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.standard_normal((8, 100)).astype("float32"))
    args = SamplingArgs(temperature=0.8, top_k=20, top_p=0.95)
    token = np.asarray(sample_logits(logits, jax.random.PRNGKey(7), args))
    assert token.shape == (8,)
    assert token.min() >= 0 and token.max() < 100


def test_sampling_rows_independent_and_seeded():
    logits = jnp.zeros((6, 200), dtype="float32")  # uniform -> spread of draws
    args = SamplingArgs(temperature=1.0, top_k=0, top_p=1.0)
    a = np.asarray(sample_logits(logits, jax.random.PRNGKey(3), args))
    b = np.asarray(sample_logits(logits, jax.random.PRNGKey(3), args))
    c = np.asarray(sample_logits(logits, jax.random.PRNGKey(4), args))
    assert np.array_equal(a, b)        # same key -> reproducible
    assert not np.array_equal(a, c)    # different key -> different draw
    assert len(set(a.tolist())) > 1    # rows are not all identical
