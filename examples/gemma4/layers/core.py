import keras
from keras import ops
from keras.layers import Layer


@keras.saving.register_keras_serializable(package="gemma4")
class MergeDims(Layer):
    """Merges two adjacent dims at call time (no parameters)."""

    def __init__(self, axis=-2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        return config

    def call(self, x):
        shape = ops.shape(x)
        idx = self.axis
        if idx < 0:
            idx = len(x.shape) + idx
        merged = shape[idx] * shape[idx + 1]
        new_shape = shape[:idx] + (merged,) + shape[idx + 2:]
        return ops.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        idx = self.axis
        if idx < 0:
            idx = len(input_shape) + idx
        a, b = input_shape[idx], input_shape[idx + 1]
        merged = None if (a is None or b is None) else a * b
        return input_shape[:idx] + (merged,) + input_shape[idx + 2:]


@keras.saving.register_keras_serializable(package="gemma4")
class SplitDim(Layer):
    """Splits one dim into two adjacent dims at call time."""

    def __init__(self, axis, sizes, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.sizes = tuple(sizes)

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["sizes"] = self.sizes
        return config

    def call(self, x):
        shape = ops.shape(x)
        idx = self.axis
        if idx < 0:
            idx = len(x.shape) + idx
        new_shape = shape[:idx] + self.sizes + shape[idx + 1:]
        return ops.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        idx = self.axis
        if idx < 0:
            idx = len(input_shape) + idx
        return input_shape[:idx] + self.sizes + input_shape[idx + 1:]


def build_attention_mask(
        padding_mask, bidirectional, sliding_window_size):
    if padding_mask is None:
        return None
    if bidirectional:
        return build_bidirectional_mask(padding_mask)
    positions = build_positions(padding_mask)
    causal_mask = compute_causal_mask(positions)
    if sliding_window_size is not None:
        window_mask = compute_sliding_window_mask(
            positions, sliding_window_size)
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


def apply_partial_rotary_embedding(
        inputs, wavelength, scaling_factor,
        partial_rotary_factor, positions=None):
    raw_dim = int(inputs.shape[-1] * partial_rotary_factor)
    rotary_dim = max(2, raw_dim - raw_dim % 2)
    if rotary_dim >= inputs.shape[-1]:
        return apply_rotary_embedding(
            inputs, wavelength, scaling_factor, positions)
    half_rotary = rotary_dim // 2
    half_head = inputs.shape[-1] // 2
    first_half = inputs[..., :half_head]
    second_half = inputs[..., half_head:]
    first_rotary = first_half[..., :half_rotary]
    second_rotary = second_half[..., :half_rotary]
    first_static = first_half[..., half_rotary:]
    second_static = second_half[..., half_rotary:]
    rotary = ops.concatenate((first_rotary, second_rotary), axis=-1)
    rotary = apply_rotary_embedding(
        rotary, wavelength, scaling_factor, positions)
    first_rotary, second_rotary = ops.split(rotary, 2, axis=-1)
    first_half = ops.concatenate(
        (first_rotary, first_static), axis=-1)
    second_half = ops.concatenate(
        (second_rotary, second_static), axis=-1)
    return ops.concatenate((first_half, second_half), axis=-1)


def apply_rotary_embedding(
        inputs, wavelength, scaling_factor, positions=None):
    cosine, sine = build_rotary_embedding(
        inputs, wavelength, scaling_factor, positions)
    first_half, second_half = ops.split(inputs, 2, axis=-1)
    rotated = ops.stack((-second_half, first_half), axis=-2)
    rotated = MergeDims(axis=-2)(rotated)
    return (inputs * cosine) + (rotated * sine)


def build_rotary_embedding(
        inputs, wavelength, scaling_factor, positions=None):
    rotary_dim = inputs.shape[-1]
    args = (rotary_dim, wavelength, scaling_factor)
    inverse = build_inverse_frequencies(*args)
    if positions is None:
        positions = build_rotary_positions(inputs)
    angles = positions * inverse
    angles = ops.tile(angles, (1, 2))
    angles = ops.expand_dims(angles, axis=0)
    if len(inputs.shape) == 4:
        angles = ops.expand_dims(angles, axis=2)
    cosine = ops.cast(ops.cos(angles), inputs.dtype)
    sine = ops.cast(ops.sin(angles), inputs.dtype)
    return cosine, sine


def build_rotary_positions(inputs):
    trailing = tuple(range(2, len(inputs.shape)))
    ones = ops.ones_like(inputs)
    if trailing:
        ones = ops.mean(ones, axis=trailing)
    ones = ops.mean(ones, axis=0, keepdims=False)
    positions = ops.cumsum(ones) - 1.0
    return ops.expand_dims(positions, axis=1)


def build_inverse_frequencies(rotary_dim, wavelength, scaling_factor):
    indices = ops.arange(0, rotary_dim, 2, dtype="float32")
    rotary_dim = ops.cast(rotary_dim, "float32")
    frequency = indices / rotary_dim
    inverse = ops.power(
        ops.cast(wavelength, "float32"), -frequency)
    scale = ops.cast(scaling_factor, "float32")
    return inverse / scale


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
