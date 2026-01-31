import keras
from keras import ops


def apply_tanh_soft_cap(values, soft_cap):
    if soft_cap is None:
        return values
    values = ops.divide(values, soft_cap)
    values = ops.tanh(values)
    return ops.multiply(values, soft_cap)


def _build_inverse_frequencies(rotary_dim, max_wavelength, scaling_factor):
    indices = ops.arange(0, rotary_dim, 2, dtype="float32")
    dim_scale = ops.cast(rotary_dim, "float32")
    frequency_range = indices / dim_scale
    power = ops.cast(max_wavelength, "float32")
    inverse = ops.power(power, -frequency_range)
    return inverse / ops.cast(scaling_factor, "float32")


def _expand_rotary_embedding(embedding, inputs):
    batch_axis = 0
    sequence_axis = 1
    feature_axis = len(inputs.shape) - 1
    for axis in range(len(inputs.shape)):
        if axis not in (batch_axis, sequence_axis, feature_axis):
            embedding = ops.expand_dims(embedding, axis)
    return embedding


def _compute_rotary_cos_sin(inputs, start, wavelength, scale):
    rotary_dim = ops.shape(inputs)[-1]
    rotary_dim = ops.cast(rotary_dim, "int32")
    inverse = _build_inverse_frequencies(rotary_dim, wavelength, scale)
    positions = ops.arange(start, start + ops.shape(inputs)[1])
    positions = ops.cast(positions, "float32")
    positions = ops.expand_dims(positions, axis=0)
    freq = ops.einsum("bi,j->bij", positions, inverse)
    embedding = ops.stack((freq, freq), axis=-2)
    freq_shape = ops.shape(freq)
    embed_dim = freq_shape[-1] * 2
    shape = (freq_shape[0], freq_shape[1], embed_dim)
    embedding = ops.reshape(embedding, shape)
    embedding = _expand_rotary_embedding(embedding, inputs)
    cos_emb = ops.cast(ops.cos(embedding), inputs.dtype)
    sin_emb = ops.cast(ops.sin(embedding), inputs.dtype)
    return cos_emb, sin_emb


def apply_rotary_embedding(inputs, start, wavelength, scale):
    cos_emb, sin_emb = _compute_rotary_cos_sin(inputs, start, wavelength, scale)
    first_half, second_half = ops.split(inputs, 2, axis=-1)
    half_rotated = ops.stack((-second_half, first_half), axis=-2)
    half_rotated = ops.reshape(half_rotated, ops.shape(inputs))
    return (inputs * cos_emb) + (half_rotated * sin_emb)


def compute_causal_mask(batch_size, input_length, output_length, cache_index=0):
    out_pos = ops.arange(output_length, dtype="float32")
    out_pos = out_pos + ops.cast(cache_index, "float32")
    out_pos = ops.expand_dims(out_pos, axis=1)
    in_pos = ops.arange(input_length, dtype="float32")
    mask = ops.expand_dims(out_pos >= in_pos, axis=0)
    shape = (batch_size, output_length, input_length)
    return ops.broadcast_to(mask, shape)


def merge_padding_and_attention_mask(padding_mask, attention_mask):
    mask = padding_mask
    if mask is not None:
        mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        return ops.minimum(mask, attention_mask)
    return mask


def clip_float16(values):
    dtype = keras.backend.standardize_dtype(values.dtype)
    if dtype == "float16":
        return ops.clip(values, -65504, 65504)
    return values


def add_residual(base, delta):
    dtype_name = keras.backend.standardize_dtype(base.dtype)
    if dtype_name != "float16":
        return base + delta
    left = ops.cast(base, "float32")
    right = ops.cast(delta, "float32")
    output = ops.add(left, right)
    output = clip_float16(output)
    return ops.cast(output, "float16")
