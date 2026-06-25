import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.models.transformers import mask


def test_causal_single_query_matches_threshold():
    query = np.array([[2]], dtype="int32")
    key = np.array([[0, 1, 2, 3, 4]], dtype="int32")
    out = np.asarray(mask.causal(query, key)).astype(bool)
    assert out.shape == (1, 1, 5)
    assert out[0, 0].tolist() == [True, True, True, False, False]


def test_causal_full_grid_is_lower_triangular():
    positions = np.array([[0, 1, 2]], dtype="int32")
    out = np.asarray(mask.causal(positions, positions)).astype(bool)
    assert out[0].tolist() == [[True, False, False],
                               [True, True, False],
                               [True, True, True]]


def test_sliding_window_keeps_recent_keys():
    positions = np.array([[0, 1, 2, 3]], dtype="int32")
    out = np.asarray(mask.sliding_window(positions, positions, 2)).astype(bool)
    # A key is kept when query - key < window. Future keys (negative distance)
    # pass too; the window mask is meant to be combined with the causal mask.
    assert out[0].tolist() == [[True, True, True, True],
                               [True, True, True, True],
                               [False, True, True, True],
                               [False, False, True, True]]


def test_sliding_window_combines_with_causal():
    positions = np.array([[0, 1, 2, 3]], dtype="int32")
    causal = np.asarray(mask.causal(positions, positions)).astype(bool)
    window = np.asarray(mask.sliding_window(positions, positions, 2))
    combined = np.logical_and(causal, window.astype(bool))
    assert combined[0, 3].tolist() == [False, False, True, True]
