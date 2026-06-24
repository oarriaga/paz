import keras
from keras import ops


def build_attention_mask(padding_mask, bidirectional, sliding_window_size):
    if padding_mask is None:
        return None
    if bidirectional:
        return build_bidirectional_mask(padding_mask)
    positions = build_positions(padding_mask)
    causal_mask = compute_causal_mask(positions)
    if sliding_window_size is not None:
        args = (positions, sliding_window_size)
        window_mask = compute_sliding_window_mask(*args)
        causal_mask = ops.logical_and(causal_mask, window_mask)
    decoder_mask = merge_padding_mask(padding_mask)
    return ops.logical_and(causal_mask, decoder_mask)


def build_positions(padding_mask):
    ones = ops.ones_like(padding_mask, dtype="int32")
    return ops.cumsum(ones, axis=1) - 1


def compute_causal_mask(positions):
    out_pos = ops.expand_dims(positions, axis=2)
    in_pos = ops.expand_dims(positions, axis=1)
    return ops.greater_equal(out_pos, in_pos)


def compute_sliding_window_mask(positions, window_size):
    out_pos = ops.expand_dims(positions, axis=2)
    in_pos = ops.expand_dims(positions, axis=1)
    distance = out_pos - in_pos
    return ops.less(distance, window_size)


def build_bidirectional_mask(padding_mask):
    if padding_mask is None:
        return None
    mask = merge_padding_mask(padding_mask)
    return ops.logical_and(mask, ops.transpose(mask, (0, 2, 1)))


def merge_padding_mask(padding_mask):
    if padding_mask is None:
        return None
    mask = ops.cast(padding_mask, "bool")
    return ops.expand_dims(mask, axis=1)


def apply_tanh_soft_cap(values, soft_cap):
    if soft_cap is None:
        return values
    values = ops.divide(values, soft_cap)
    values = ops.tanh(values)
    return ops.multiply(values, soft_cap)


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
