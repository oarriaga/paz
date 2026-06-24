"""Rotary position embedding (RoPE)."""
from keras import ops

from paz.layers import MergeDims


def apply(inputs, wavelength, scaling_factor, denominator, positions=None):
    cosine, sine = build(inputs, wavelength, scaling_factor, denominator,
                         positions)
    first_half, second_half = ops.split(inputs, 2, axis=-1)
    rotated = ops.stack((-second_half, first_half), axis=-2)
    rotated = MergeDims(axis=-2)(rotated)
    return (inputs * cosine) + (rotated * sine)


def apply_partial(inputs, wavelength, scaling_factor, partial_rotary_factor,
                  positions=None):
    head_dim = inputs.shape[-1]
    raw_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, raw_dim - raw_dim % 2)
    if rotary_dim >= head_dim:
        return apply(inputs, wavelength, scaling_factor, head_dim, positions)
    half_rotary = rotary_dim // 2
    half_head = head_dim // 2
    first_half = inputs[..., :half_head]
    second_half = inputs[..., half_head:]
    first_rotary = first_half[..., :half_rotary]
    second_rotary = second_half[..., :half_rotary]
    first_static = first_half[..., half_rotary:]
    second_static = second_half[..., half_rotary:]
    rotary = ops.concatenate((first_rotary, second_rotary), axis=-1)
    rotary = apply(rotary, wavelength, scaling_factor, head_dim, positions)
    first_rotary, second_rotary = ops.split(rotary, 2, axis=-1)
    first_half = ops.concatenate((first_rotary, first_static), axis=-1)
    second_half = ops.concatenate((second_rotary, second_static), axis=-1)
    return ops.concatenate((first_half, second_half), axis=-1)


def build(inputs, wavelength, scaling_factor, denominator, positions=None):
    rotary_dim = inputs.shape[-1]
    args = (rotary_dim, denominator, wavelength, scaling_factor)
    inverse = build_frequencies(*args)
    if positions is None:
        positions = build_positions(inputs)
    angles = positions * inverse
    angles = ops.tile(angles, (1, 2))
    angles = ops.expand_dims(angles, axis=0)
    if len(inputs.shape) == 4:
        angles = ops.expand_dims(angles, axis=2)
    cosine = ops.cast(ops.cos(angles), inputs.dtype)
    sine = ops.cast(ops.sin(angles), inputs.dtype)
    return cosine, sine


def build_positions(inputs):
    trailing = tuple(range(2, len(inputs.shape)))
    ones = ops.ones_like(inputs)
    if trailing:
        ones = ops.mean(ones, axis=trailing)
    ones = ops.mean(ones, axis=0, keepdims=False)
    positions = ops.cumsum(ones) - 1.0
    return ops.expand_dims(positions, axis=1)


def build_frequencies(rotary_dim, denominator, wavelength, scaling_factor):
    indices = ops.arange(0, rotary_dim, 2, dtype="float32")
    denominator = ops.cast(denominator, "float32")
    frequency = indices / denominator
    inverse = ops.power(ops.cast(wavelength, "float32"), -frequency)
    scale = ops.cast(scaling_factor, "float32")
    return inverse / scale
