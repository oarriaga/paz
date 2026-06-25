"""Numeric helpers for low-precision (float16) transformer arithmetic.

Float16 overflows at +-65504, so residual sums are accumulated in float32 and
clipped before casting back. Models running in float32 or bfloat16 pass through
unchanged.
"""
import keras
from keras import ops


def clip_float16(values):
    dtype = keras.backend.standardize_dtype(values.dtype)
    if dtype != "float16":
        return values
    return ops.clip(values, -65504, 65504)


def add_residual(left, right):
    dtype = keras.backend.standardize_dtype(left.dtype)
    if dtype != "float16":
        return left + right
    left = ops.cast(left, "float32")
    right = ops.cast(right, "float32")
    output = clip_float16(ops.add(left, right))
    return ops.cast(output, "float16")
