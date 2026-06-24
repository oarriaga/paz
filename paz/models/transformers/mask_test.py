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
