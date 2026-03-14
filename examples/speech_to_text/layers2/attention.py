import string

import keras
from keras import ops


def split_last_dimension_into_heads(tensor, num_heads, head_size):
    batch_size = ops.shape(tensor)[0]
    sequence_length = ops.shape(tensor)[1]
    tensor = ops.reshape(
        tensor,
        (batch_size, sequence_length, num_heads, head_size),
    )
    return ops.transpose(tensor, (0, 2, 1, 3))


def merge_heads_into_last_dimension(tensor):
    return ops.transpose(tensor, (0, 2, 1, 3))


def build_padding_attention_mask(padding_mask, attention_mask=None):
    mask = None
    if padding_mask is not None:
        mask = ops.cast(ops.expand_dims(padding_mask, axis=1), "int32")
    if attention_mask is None:
        return mask
    attention_mask = ops.cast(attention_mask, "int32")
    if mask is None:
        return attention_mask
    return ops.minimum(mask, attention_mask)


def build_causal_attention_mask(query, value):
    query_positions = ops.ones_like(query[..., 0], dtype="int32")
    key_positions = ops.ones_like(value[..., 0], dtype="int32")
    query_positions = ops.cumsum(query_positions, axis=1)
    key_positions = ops.cumsum(key_positions, axis=1)
    return ops.cast(
        ops.expand_dims(query_positions, axis=2)
        >= ops.expand_dims(key_positions, axis=1),
        "int32",
    )


def build_decoder_self_attention_mask(decoder_sequence, decoder_padding_mask):
    causal_mask = build_causal_attention_mask(
        decoder_sequence,
        decoder_sequence,
    )
    return build_padding_attention_mask(decoder_padding_mask, causal_mask)


def expand_attention_mask_for_heads(attention_mask):
    if attention_mask is None:
        return None
    attention_mask = ops.cast(attention_mask, "bool")
    if len(attention_mask.shape) == 2:
        attention_mask = ops.expand_dims(attention_mask, axis=1)
        attention_mask = ops.expand_dims(attention_mask, axis=1)
    elif len(attention_mask.shape) == 3:
        attention_mask = ops.expand_dims(attention_mask, axis=1)
    return attention_mask


def build_combined_attention_mask(
    attention_mask, use_causal_mask, query, value
):
    padding_mask = expand_attention_mask_for_heads(attention_mask)
    if not use_causal_mask:
        return padding_mask
    causal_mask = build_causal_attention_mask(query, value)
    causal_mask = expand_attention_mask_for_heads(causal_mask)
    if padding_mask is None:
        return causal_mask
    return ops.logical_and(padding_mask, causal_mask)


def apply_attention_mask_to_scores(attention_scores, attention_mask):
    if attention_mask is None:
        return attention_scores
    large_negative_value = ops.cast(-1e9, attention_scores.dtype)
    return ops.where(attention_mask, attention_scores, large_negative_value)


def index_to_einsum_variable(index):
    return string.ascii_lowercase[index]


def build_projection_equation(free_dims, bound_dims, output_dims):
    input_string = ""
    kernel_string = ""
    output_string = ""
    bias_axes = ""
    letter_offset = 0
    for index in range(free_dims):
        character = index_to_einsum_variable(index + letter_offset)
        input_string += character
        output_string += character
    letter_offset += free_dims
    for index in range(bound_dims):
        character = index_to_einsum_variable(index + letter_offset)
        input_string += character
        kernel_string += character
    letter_offset += bound_dims
    for index in range(output_dims):
        character = index_to_einsum_variable(index + letter_offset)
        kernel_string += character
        output_string += character
        bias_axes += character
    equation = f"{input_string},{kernel_string}->{output_string}"
    return equation, bias_axes, len(output_string)


def build_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(
        known_last_dims
    )


def build_projection(
    inputs,
    free_dims,
    bound_dims,
    output_dims,
    output_shape,
    use_bias,
    kernel_initializer,
    bias_initializer,
    dtype,
    name,
):
    equation, bias_axes, output_rank = build_projection_equation(
        free_dims,
        bound_dims,
        output_dims,
    )
    projection = keras.layers.EinsumDense(
        equation,
        output_shape=build_output_shape(output_rank - 1, output_shape),
        bias_axes=bias_axes if use_bias else None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        dtype=dtype,
        name=name,
    )
    return projection(inputs)


def attention(
    query,
    value,
    key=None,
    attention_mask=None,
    num_heads=8,
    key_dim=64,
    value_dim=None,
    dropout=0.0,
    use_bias=True,
    use_causal_mask=False,
    kernel_initializer=None,
    bias_initializer="zeros",
    dtype="float32",
    name="attention",
):
    if key is None:
        key = value
    if value_dim is None:
        value_dim = key_dim
    if kernel_initializer is None:
        kernel_initializer = keras.initializers.TruncatedNormal(stddev=0.02)

    query_rank = len(query.shape)
    value_rank = len(value.shape)
    key_rank = len(key.shape)
    output_dim = query.shape[-1]

    query_projection = build_projection(
        query,
        query_rank - 1,
        1,
        2,
        [num_heads, key_dim],
        use_bias,
        kernel_initializer,
        bias_initializer,
        dtype,
        f"{name}_query",
    )
    key_projection = build_projection(
        key,
        key_rank - 1,
        1,
        2,
        [num_heads, key_dim],
        False,
        kernel_initializer,
        bias_initializer,
        dtype,
        f"{name}_key",
    )
    value_projection = build_projection(
        value,
        value_rank - 1,
        1,
        2,
        [num_heads, value_dim],
        use_bias,
        kernel_initializer,
        bias_initializer,
        dtype,
        f"{name}_value",
    )

    query_heads = ops.transpose(query_projection, (0, 2, 1, 3))
    key_heads = ops.transpose(key_projection, (0, 2, 1, 3))
    value_heads = ops.transpose(value_projection, (0, 2, 1, 3))
    scaling_factor = ops.sqrt(ops.cast(key_dim, query_heads.dtype))
    query_heads = query_heads / scaling_factor
    key_heads = ops.transpose(key_heads, (0, 1, 3, 2))
    attention_scores = ops.matmul(query_heads, key_heads)
    attention_mask = build_combined_attention_mask(
        attention_mask,
        use_causal_mask,
        query,
        value,
    )
    attention_scores = apply_attention_mask_to_scores(
        attention_scores, attention_mask
    )
    attention_probabilities = ops.softmax(attention_scores, axis=-1)
    attention_probabilities = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name=f"{name}_attention_scores_dropout",
    )(attention_probabilities)
    attention_values = ops.matmul(attention_probabilities, value_heads)
    attention_values = merge_heads_into_last_dimension(attention_values)
    return build_projection(
        attention_values,
        query_rank - 1,
        2,
        1,
        [output_dim],
        use_bias,
        kernel_initializer,
        bias_initializer,
        dtype,
        f"{name}_attention_output",
    )
