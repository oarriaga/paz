import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
from keras import ops

from paz.models.transformers import numerics


def test_clip_float16_passes_through_float32():
    values = ops.convert_to_tensor([1e6, -1e6], dtype="float32")
    out = np.asarray(numerics.clip_float16(values))
    assert np.allclose(out, [1e6, -1e6])


def test_clip_float16_clips_float16_overflow():
    values = ops.convert_to_tensor([70000.0, -70000.0], dtype="float16")
    out = np.asarray(numerics.clip_float16(values)).astype("float32")
    assert np.allclose(out, [65504.0, -65504.0])


def test_add_residual_float32_is_plain_sum():
    left = ops.convert_to_tensor([1.0, 2.0], dtype="float32")
    right = ops.convert_to_tensor([3.0, 4.0], dtype="float32")
    out = np.asarray(numerics.add_residual(left, right))
    assert np.allclose(out, [4.0, 6.0])


def test_add_residual_float16_accumulates_in_float32():
    left = ops.convert_to_tensor([1000.0], dtype="float16")
    right = ops.convert_to_tensor([2000.0], dtype="float16")
    result = numerics.add_residual(left, right)
    assert np.allclose(np.asarray(result).astype("float32"), [3000.0])
