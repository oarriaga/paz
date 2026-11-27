import numpy as np
import pytest
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


REASON = ("add_residual clips the float32 sum with clip_float16, which "
          "skips float32 inputs, so the cast back to float16 gives inf")


@pytest.mark.xfail(strict=True, reason=REASON)
def test_add_residual_float16_clips_a_sum_that_would_overflow():
    # 60000 + 60000 is infinite in float16; the float32 sum is clipped.
    left = ops.convert_to_tensor([60000.0, 1000.0], dtype="float16")
    right = ops.convert_to_tensor([60000.0, 2000.0], dtype="float16")
    result = numerics.add_residual(left, right)
    assert result.dtype == "float16"
    values = np.asarray(result).astype("float32")
    assert np.allclose(values, [65504.0, 3000.0])
