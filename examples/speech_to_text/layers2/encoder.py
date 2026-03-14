import keras

from examples.speech_to_text.layers2.attention import attention


def kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


def self_attention_block(
    inputs,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    residual = inputs
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name="self_attention_layer_norm",
    )(inputs)
    hidden = attention(
        hidden,
        hidden,
        None,
        None,
        num_heads,
        key_dim,
        None,
        dropout,
        True,
        False,
        kernel_initializer(),
        "zeros",
        dtype,
        "self_attention_layer",
    )
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name="self_attention_dropout",
    )(hidden)
    return hidden + residual


def feedforward_block(
    inputs,
    intermediate_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name="feedforward_layer_norm",
    )(inputs)
    hidden = keras.layers.Dense(
        intermediate_dim,
        activation=keras.activations.gelu,
        kernel_initializer=kernel_initializer(),
        bias_initializer="zeros",
        dtype=dtype,
        name="feedforward_intermediate_dense",
    )(hidden)
    hidden = keras.layers.Dense(
        inputs.shape[-1],
        kernel_initializer=kernel_initializer(),
        bias_initializer="zeros",
        dtype=dtype,
        name="feedforward_output_dense",
    )(hidden)
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name="feedforward_dropout",
    )(hidden)
    return hidden + inputs


def encoder_block(
    inputs,
    num_heads,
    intermediate_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    hidden_dim = inputs.shape[-1]
    if hidden_dim is None:
        raise ValueError("Encoder block inputs must have known hidden size.")
    key_dim = int(hidden_dim // num_heads)
    hidden = self_attention_block(
        inputs,
        num_heads,
        key_dim,
        dropout,
        epsilon,
        dtype,
    )
    return feedforward_block(
        hidden,
        intermediate_dim,
        dropout,
        epsilon,
        dtype,
    )
