import keras
from keras import ops
from keras.layers import Dense, Dropout
from paz.models.foundation.dinov3.utils.utils import cat_keep_shapes, uncat_with_shapes


def compute_multihead_self_attention(qkv, num_heads, head_dim, dim, scale, dropout_layer, rope, training):
    batch_size, num_tokens, _ = ops.shape(qkv)
    queries, keys, values = split_qkv_into_queries_keys_values(qkv, num_heads, head_dim)
    queries = transpose_for_multihead_attention(queries)
    keys = transpose_for_multihead_attention(keys)
    values = transpose_for_multihead_attention(values)

    if rope is not None:
        queries, keys = apply_rope_to_queries_and_keys(queries, keys, rope)

    attended_values = compute_attention_output(queries, keys, values, scale, dropout_layer, training)
    return merge_attention_heads(attended_values, batch_size, num_tokens, dim)


def compute_causal_multihead_self_attention(
    qkv, num_heads, head_dim, dim, scale, dropout_layer, is_causal, training
):
    batch_size, num_tokens, _ = ops.shape(qkv)
    queries, keys, values = split_qkv_into_queries_keys_values(qkv, num_heads, head_dim)
    queries = transpose_for_multihead_attention(queries)
    keys = transpose_for_multihead_attention(keys)
    values = transpose_for_multihead_attention(values)

    attention_scores = compute_scaled_attention_scores(queries, keys, scale)

    if is_causal:
        sequence_length = ops.shape(queries)[2]
        causal_mask = create_causal_attention_mask(sequence_length)
        attention_scores = attention_scores + causal_mask

    attention_weights = ops.softmax(attention_scores, axis=-1)
    attention_weights = dropout_layer(attention_weights, training=training)

    attended_values = apply_attention_to_values(attention_weights, values)
    return merge_attention_heads(attended_values, batch_size, num_tokens, dim)


def compute_attention_output(queries, keys, values, scale, dropout_layer, training):
    attention_scores = compute_scaled_attention_scores(queries, keys, scale)
    attention_weights = ops.softmax(attention_scores, axis=-1)
    attention_weights = dropout_layer(attention_weights, training=training)
    attended_values = apply_attention_to_values(attention_weights, values)
    return attended_values


def apply_rope_to_queries_and_keys(queries, keys, rope):
    sin, cos = rope
    rope_sequence_length = ops.shape(sin)[-2]

    queries_prefix, queries_main, keys_prefix, keys_main = split_queries_and_keys_for_rope(
        queries, keys, rope_sequence_length
    )

    queries_rotated = apply_rotary_position_embedding(queries_main, sin, cos)
    keys_rotated = apply_rotary_position_embedding(keys_main, sin, cos)

    queries = ops.concatenate((queries_prefix, queries_rotated), axis=-2)
    keys = ops.concatenate((keys_prefix, keys_rotated), axis=-2)

    return queries, keys


def apply_projection_with_dropout(x, projection_layer, dropout_layer, training):
    projected = projection_layer(x)
    return dropout_layer(projected, training=training)


def split_qkv_into_queries_keys_values(qkv, num_heads, head_dim):
    batch_size, num_tokens, _ = ops.shape(qkv)
    qkv_reshaped = ops.reshape(qkv, (batch_size, num_tokens, 3, num_heads, head_dim))
    queries, keys, values = ops.unstack(qkv_reshaped, axis=2)
    return queries, keys, values


def transpose_for_multihead_attention(tensor):
    return ops.transpose(tensor, axes=[0, 2, 1, 3])


def compute_scaled_attention_scores(queries, keys, scale):
    keys_transposed = ops.transpose(keys, axes=[0, 1, 3, 2])
    attention_scores = ops.matmul(queries, keys_transposed)
    return attention_scores * scale


def apply_attention_to_values(attention_weights, values):
    return ops.matmul(attention_weights, values)


def merge_attention_heads(tensor, batch_size, num_tokens, dim):
    tensor_transposed = ops.transpose(tensor, axes=[0, 2, 1, 3])
    return ops.reshape(tensor_transposed, (batch_size, num_tokens, dim))


def create_causal_attention_mask(sequence_length):
    mask = ops.triu(ops.ones((sequence_length, sequence_length)), k=1)
    return mask * -1e9


def split_queries_and_keys_for_rope(queries, keys, rope_sequence_length):
    num_prefix_tokens = ops.shape(queries)[-2] - rope_sequence_length

    if num_prefix_tokens < 0:
        raise ValueError("Input sequence is shorter than the RoPE sequence.")

    queries_prefix = queries[:, :, :num_prefix_tokens, :]
    queries_main = queries[:, :, num_prefix_tokens:, :]
    keys_prefix = keys[:, :, :num_prefix_tokens, :]
    keys_main = keys[:, :, num_prefix_tokens:, :]

    return queries_prefix, queries_main, keys_prefix, keys_main


def apply_rotary_position_embedding(x, sin, cos):
    return (x * cos) + (rotate_half_hidden_dimensions(x) * sin)


def rotate_half_hidden_dimensions(x):
    first_half, second_half = keras.ops.split(x, 2, axis=-1)
    return keras.ops.concatenate([-second_half, first_half], axis=-1)


class SelfAttention(keras.Layer):
    """Keras implementation of DINOv3's SelfAttention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop_layer = Dropout(attn_drop)
        self.proj = Dense(dim, use_bias=proj_bias, name="proj")
        self.proj_drop_layer = Dropout(proj_drop)

    def apply_rope(self, queries, keys, rope):
        return apply_rope_to_queries_and_keys(queries, keys, rope)

    def compute_attention(self, qkv, rope=None, training=None):
        return compute_multihead_self_attention(
            qkv, self.num_heads, self.head_dim, self.dim,
            self.scale, self.attn_drop_layer, rope, training
        )

    def forward_tensor(self, x, rope=None, training=None):
        qkv = self.qkv(x)
        attention_output = self.compute_attention(qkv=qkv, rope=rope, training=training)
        return apply_projection_with_dropout(
            attention_output, self.proj, self.proj_drop_layer, training
        )

    def forward_list(self, x_list, rope_list=None, training=None):
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)

        if rope_list is None:
            rope_list = [None] * len(qkv_list)

        attention_outputs = [
            self.compute_attention(qkv, rope=rope, training=training)
            for qkv, rope in zip(qkv_list, rope_list)
        ]

        x_flat_out, shapes_out, num_tokens_out = cat_keep_shapes(attention_outputs)
        output_with_dropout = apply_projection_with_dropout(
            x_flat_out, self.proj, self.proj_drop_layer, training
        )

        return uncat_with_shapes(output_with_dropout, shapes_out, num_tokens_out)

    def call(self, x, rope=None, training=None):
        if isinstance(x, list):
            return self.forward_list(x, rope_list=rope, training=training)
        else:
            return self.forward_tensor(x, rope=rope, training=training)


class CausalSelfAttention(keras.Layer):
    """Keras implementation of DINOv3's CausalSelfAttention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop_layer = Dropout(attn_drop)
        self.proj = Dense(dim, use_bias=proj_bias, name="proj")
        self.proj_drop_layer = Dropout(proj_drop)

    def call(self, x, is_causal=True, training=None):
        qkv = self.qkv(x)
        attention_output = compute_causal_multihead_self_attention(
            qkv, self.num_heads, self.head_dim, self.dim,
            self.scale, self.attn_drop_layer, is_causal, training
        )
        return apply_projection_with_dropout(
            attention_output, self.proj, self.proj_drop_layer, training
        )
