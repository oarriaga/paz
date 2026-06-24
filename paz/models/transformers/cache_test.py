import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
from keras import ops

from paz.models.transformers import cache


def test_build_shape_and_zeros():
    state = cache.build(1, 3, 5, 4, 2)
    assert tuple(state.shape) == (1, 3, 2, 5, 4, 2)
    assert float(ops.max(ops.abs(state))) == 0.0


def test_update_writes_key_and_value_at_index():
    state = ops.zeros((1, 2, 4, 3, 2))
    key = ops.ones((1, 1, 3, 2))
    value = ops.full((1, 1, 3, 2), 2.0)
    out = np.asarray(cache.update(state, 1, key, value))
    assert np.all(out[:, 0, 1] == 1.0)
    assert np.all(out[:, 1, 1] == 2.0)
    assert np.all(out[:, :, 0] == 0.0) and np.all(out[:, :, 2] == 0.0)
