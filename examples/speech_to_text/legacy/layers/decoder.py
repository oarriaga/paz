import keras

from .attention import attention
from .attention import cached_attention
from .attention import build_decoder_self_attention_mask
from .attention import build_padding_attention_mask
from .encoder import kernel_initializer


def self_attention_block(
    decoder_sequence,
    decoder_padding_mask,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="self_attention",
):
    residual = decoder_sequence
    attention_mask = build_decoder_self_attention_mask(
        decoder_sequence,
        decoder_padding_mask,
    )
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=f"{name}_layer_norm",
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
        name,
    )
    hidden = keras.layers.Dropout(
        dropout,
        dtype=dtype,
        name=f"{name}_dropout",
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
    name="cross_attention",
):
    residual = decoder_sequence
    attention_mask = build_padding_attention_mask(encoder_padding_mask)
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=f"{name}_layer_norm",
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
    residual = inputs
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
    name="transformer_decoder_layer",
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
        f"{name}_self_attention",
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
        f"{name}_cross_attention",
    )
    return feedforward_block(
        hidden,
        intermediate_dim,
        dropout,
        epsilon,
        dtype,
        f"{name}_feedforward",
    )


def build_cached_self_attention_mask(self_attention_cache, cache_update_index):
    valid_positions = keras.ops.ones_like(
        self_attention_cache[:, 0, :, 0, 0], dtype="int32"
    )
    valid_positions = keras.ops.cumsum(valid_positions, axis=1) - 1
    return keras.ops.cast(
        keras.ops.expand_dims(valid_positions, axis=1)
        <= cache_update_index,
        "int32",
    )


def cached_self_attention_block(
    decoder_sequence,
    self_attention_cache,
    cache_update_index,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="self_attention",
):
    residual = decoder_sequence
    attention_mask = build_cached_self_attention_mask(
        self_attention_cache, cache_update_index
    )
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=f"{name}_layer_norm",
    )(decoder_sequence)
    hidden, self_attention_cache = cached_attention(
        hidden,
        self_attention_cache,
        cache_update_index,
        hidden,
        None,
        attention_mask,
        num_heads,
        key_dim,
        None,
        dropout,
        True,
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
    return hidden + residual, self_attention_cache


def cached_cross_attention_block(
    decoder_sequence,
    cross_attention_cache,
    num_heads,
    key_dim,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="cross_attention",
):
    residual = decoder_sequence
    hidden = keras.layers.LayerNormalization(
        epsilon=epsilon,
        dtype=dtype,
        name=f"{name}_layer_norm",
    )(decoder_sequence)
    hidden, _ = cached_attention(
        hidden,
        cross_attention_cache,
        None,
        None,
        None,
        None,
        num_heads,
        key_dim,
        key_dim,
        dropout,
        True,
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


def cached_decoder_block(
    decoder_sequence,
    self_attention_cache,
    cross_attention_cache,
    cache_update_index,
    num_heads=8,
    intermediate_dim=2048,
    dropout=0.0,
    epsilon=1e-5,
    dtype="float32",
    name="transformer_decoder_layer",
):
    hidden_dim = decoder_sequence.shape[-1]
    if hidden_dim is None:
        raise ValueError("Decoder block inputs must have known hidden size.")
    key_dim = int(hidden_dim // num_heads)
    hidden, self_attention_cache = cached_self_attention_block(
        decoder_sequence,
        self_attention_cache,
        cache_update_index,
        num_heads,
        key_dim,
        dropout,
        epsilon,
        dtype,
        f"{name}_self_attention",
    )
    hidden = cached_cross_attention_block(
        hidden,
        cross_attention_cache,
        num_heads,
        key_dim,
        dropout,
        epsilon,
        dtype,
        f"{name}_cross_attention",
    )
    hidden = feedforward_block(
        hidden,
        intermediate_dim,
        dropout,
        epsilon,
        dtype,
        f"{name}_feedforward",
    )
    return hidden, self_attention_cache
