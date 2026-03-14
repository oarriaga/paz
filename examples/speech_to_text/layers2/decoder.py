import keras

from examples.speech_to_text.layers2.attention import attention
from examples.speech_to_text.layers2.attention import build_decoder_self_attention_mask
from examples.speech_to_text.layers2.attention import build_padding_attention_mask
from examples.speech_to_text.layers2.encoder import kernel_initializer


def self_attention_block(
    decoder_sequence,
    decoder_padding_mask,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    residual = decoder_sequence
    attention_mask = build_decoder_self_attention_mask(
        decoder_sequence,
        decoder_padding_mask,
    )
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name="self_attention_layer_norm",
    )(decoder_sequence)
    hidden = attention(
        hidden,
        hidden,
        None,
        attention_mask,
        num_heads,
        key_dim,
        None,
        dropout,
        True,
        False,
        kernel_initializer(),
        "zeros",
        dtype,
        "self_attention",
    )
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name="self_attention_dropout",
    )(hidden)
    return hidden + residual


def cross_attention_block(
    decoder_sequence,
    encoder_sequence,
    encoder_padding_mask,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    residual = decoder_sequence
    attention_mask = build_padding_attention_mask(encoder_padding_mask)
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name="cross_attention_layer_norm",
    )(decoder_sequence)
    hidden = attention(
        hidden,
        encoder_sequence,
        None,
        attention_mask,
        num_heads,
        key_dim,
        key_dim,
        dropout,
        True,
        False,
        kernel_initializer(),
        "zeros",
        dtype,
        "cross_attention",
    )
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name="cross_attention_dropout",
    )(hidden)
    return hidden + residual


def feedforward_block(
    inputs,
    intermediate_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    residual = inputs
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
    return hidden + residual


def decoder_block(
    decoder_sequence,
    encoder_sequence,
    decoder_padding_mask,
    encoder_padding_mask=None,
    num_heads=8,
    intermediate_dim=2048,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
):
    hidden_dim = decoder_sequence.shape[-1]
    if hidden_dim is None:
        raise ValueError("Decoder block inputs must have known hidden size.")
    key_dim = int(hidden_dim // num_heads)
    hidden = self_attention_block(
        decoder_sequence,
        decoder_padding_mask,
        num_heads,
        key_dim,
        dropout,
        epsilon,
        dtype,
    )
    hidden = cross_attention_block(
        hidden,
        encoder_sequence,
        encoder_padding_mask,
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
