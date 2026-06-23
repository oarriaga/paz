import keras
from keras import ops
from keras.layers import Dense, Layer

from .attention import (AttendArgs, compute_attention, project_key,
                        project_output, project_query, project_value)
from .decoder import build_einsum_dense
from .normalization import build_rms_norm, build_v_norm


@keras.saving.register_keras_serializable(package="gemma4")
class PositionEmbeddingTable(Layer):
    def __init__(self, count, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.hidden_dim = hidden_dim

    def get_config(self):
        config = super().get_config()
        config["count"] = self.count
        config["hidden_dim"] = self.hidden_dim
        return config

    def build(self, input_shape):
        self.table = self.add_weight(
            name="position_embedding_table",
            shape=(2, self.count, self.hidden_dim),
            initializer="ones",
            trainable=True,
        )
        self.built = True

    def call(self, position_ids):
        clamped = ops.maximum(position_ids, 0)
        x_embeds = ops.take(self.table[0], clamped[..., 0], axis=0)
        y_embeds = ops.take(self.table[1], clamped[..., 1], axis=0)
        embeds = x_embeds + y_embeds
        is_padding = ops.all(
            ops.equal(position_ids, -1), axis=-1, keepdims=True)
        return embeds * (1.0 - ops.cast(is_padding, embeds.dtype))


def build_patch_embedder(pixel_values, position_ids, config):
    pixel_values = 2.0 * (pixel_values - 0.5)
    projection = Dense(
        config.hidden_dim, use_bias=False, dtype=config.dtype,
        name="input_proj")
    table = PositionEmbeddingTable(
        config.position_embedding_size, config.hidden_dim,
        dtype=config.dtype, name="position_embedding_table")
    return projection(pixel_values) + table(position_ids)


def vision_decoder_block(x, mask, position_ids, config, name):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    residual = x
    norm_name = "{}_pre_attention_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, norm_name)(x)
    hidden = vision_attend(hidden, mask, position_ids, config, name)
    post_name = "{}_post_attention_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, post_name)(hidden)
    x = residual + hidden
    residual = x
    pre_ffw_name = "{}_pre_ffw_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, pre_ffw_name)(x)
    hidden = vision_feedforward(hidden, config, name)
    post_ffw_name = "{}_post_ffw_norm".format(name)
    hidden = build_rms_norm(epsilon, dtype, post_ffw_name)(hidden)
    return residual + hidden


def vision_attend(x, mask, position_ids, config, name):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    attn_name = "{}_attention".format(name)
    args = build_vision_attend_args(mask, config, attn_name)
    query = project_query(
        x, config.num_heads, config.head_dim, epsilon, dtype, attn_name)
    key = project_key(
        x, config.num_key_value_heads, config.head_dim, epsilon, dtype,
        attn_name)
    value = project_value(
        x, config.num_key_value_heads, config.head_dim, epsilon, dtype,
        attn_name)
    query = apply_vision_rotary_embedding(
        query, position_ids, config.rope_wavelength)
    key = apply_vision_rotary_embedding(
        key, position_ids, config.rope_wavelength)
    output = compute_attention(query, key, value, mask, args)
    return project_output(output, x.shape[-1], dtype, attn_name)


def build_vision_attend_args(mask, config, name):
    return AttendArgs(
        None, mask, config.head_dim, config.num_heads,
        config.num_key_value_heads, config.layer_norm_epsilon,
        config.rope_wavelength, 1.0, 1.0, None, config.dropout,
        config.dtype, name)


def vision_feedforward(x, config, name):
    dtype = config.dtype
    gate_name = "{}_ffw_gating".format(name)
    gate = build_einsum_dense(
        "btd,df->btf", config.intermediate_dim, dtype, gate_name)(x)
    value_name = "{}_ffw_gating_2".format(name)
    value = build_einsum_dense(
        "btd,df->btf", config.intermediate_dim, dtype, value_name)(x)
    hidden = keras.activations.gelu(gate, approximate=True) * value
    linear_name = "{}_ffw_linear".format(name)
    return build_einsum_dense(
        "btf,fd->btd", config.hidden_dim, dtype, linear_name)(hidden)


def apply_vision_rotary_embedding(inputs, position_ids, wavelength):
    half_head = inputs.shape[-1] // 2
    first_half = inputs[..., :half_head]
    second_half = inputs[..., half_head:]
    x_rotated = rotate_axis(first_half, position_ids[..., 0], wavelength)
    y_rotated = rotate_axis(second_half, position_ids[..., 1], wavelength)
    return ops.concatenate((x_rotated, y_rotated), axis=-1)


def rotate_axis(part, ids, wavelength):
    dim = part.shape[-1]
    index = ops.arange(0, dim, 2, dtype="float32")
    inverse = ops.power(ops.cast(wavelength, "float32"), -index / dim)
    angles = ops.einsum("bi,j->bij", ops.cast(ids, "float32"), inverse)
    angles = ops.concatenate((angles, angles), axis=-1)
    cosine = ops.cast(ops.expand_dims(ops.cos(angles), axis=2), part.dtype)
    sine = ops.cast(ops.expand_dims(ops.sin(angles), axis=2), part.dtype)
    first, second = ops.split(part, 2, axis=-1)
    rotated = ops.concatenate((-second, first), axis=-1)
    return part * cosine + rotated * sine


def build_vision_attention_mask(position_ids):
    # Mask only keys (padding patches); queries stay unmasked, matching HF.
    # Shape (batch, 1, keys) broadcasts over query positions in the softmax.
    valid = ops.any(ops.not_equal(position_ids, -1), axis=-1)
    return ops.expand_dims(valid, axis=1)


def build_real_patch_mask(position_ids):
    valid = ops.any(ops.not_equal(position_ids, -1), axis=-1, keepdims=True)
    return ops.cast(valid, "float32")


def build_average_pooling(hidden, position_ids, config):
    # hidden (images, max_patches, dim). The pooled-grid width is derived from
    # the patch positions, so rectangular/padded images pool correctly.
    k = config.pool_size
    pooled_length = config.max_patches // (k * k)
    clamped = ops.maximum(position_ids, 0)
    kernel_x = clamped[..., 0] // k
    kernel_y = clamped[..., 1] // k
    width = (ops.max(clamped[..., 0]) + 1) // k
    kernel_index = kernel_x + width * kernel_y
    is_padding = ops.all(ops.equal(position_ids, -1), axis=-1, keepdims=True)
    hidden = hidden * (1.0 - ops.cast(is_padding, hidden.dtype))
    weights = ops.cast(ops.one_hot(kernel_index, pooled_length), hidden.dtype)
    weights = weights / ops.cast(k * k, hidden.dtype)
    pooled = ops.matmul(ops.transpose(weights, (0, 2, 1)), hidden)
    scale = ops.cast(float(config.hidden_dim) ** 0.5, hidden.dtype)
    return pooled * scale


def build_vision_output(hidden, config):
    epsilon, dtype = config.layer_norm_epsilon, config.dtype
    hidden = build_v_norm(epsilon, dtype, "output_norm")(hidden)
    return Dense(
        config.output_dim, use_bias=False, dtype=dtype,
        name="vision_input_projection")(hidden)
