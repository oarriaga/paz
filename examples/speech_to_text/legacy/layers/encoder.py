import keras

from .attention import attention


def kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


def self_attention_block(
    inputs,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="self_attention_layer",
):
    residual = inputs
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=f"{name}_norm",
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
        name,
    )
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name=f"{name}_dropout",
    )(hidden)
    return hidden + residual


def feedforward_block(
    inputs,
    intermediate_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="feedforward",
):
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=f"{name}_layer_norm",
    )(inputs)
    hidden = keras.layers.Dense(
        intermediate_dim,
        activation=keras.activations.gelu,
        kernel_initializer=kernel_initializer(),
        bias_initializer="zeros",
        dtype=dtype,
        name=f"{name}_intermediate_dense",
    )(hidden)
    hidden = keras.layers.Dense(
        inputs.shape[-1],
        kernel_initializer=kernel_initializer(),
        bias_initializer="zeros",
        dtype=dtype,
        name=f"{name}_output_dense",
    )(hidden)
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name=f"{name}_dropout",
    )(hidden)
    return hidden + inputs


def encoder_block(
    inputs,
    num_heads,
    intermediate_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="transformer_encoder_layer",
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
        f"{name}_self_attention_layer",
    )
    return feedforward_block(
        hidden,
        intermediate_dim,
        dropout,
        epsilon,
        dtype,
        f"{name}_feedforward",
    )
