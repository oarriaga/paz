from keras import ops
from keras.layers import LayerNormalization, Dropout

from examples.speech_to_text.layers.utils import build_gelu_dense, build_dense
from examples.speech_to_text.layers.attention import kv_attend


def decoder_block(x, caches, index, config, name):
    self_cache, cross_cache = caches
    self_name = f"{name}_self_attention"
    hidden, self_cache = attend_self(x, self_cache, index, config, self_name)
    cross_name = f"{name}_cross_attention"
    hidden = attend_cross(hidden, cross_cache, config, cross_name)
    hidden = feedforward(hidden, config, f"{name}_feedforward")
    return hidden, self_cache


def attend_self(x, cache, index, config, name):
    mask = build_self_attention_mask(cache, index)
    norm_name = f"{name}_layer_norm"
    hidden = LayerNormalization(epsilon=1e-5, name=norm_name)(x)
    args = (hidden, cache, index, hidden, mask, config, name)
    hidden, cache = kv_attend(*args)
    hidden = Dropout(config["dropout"], name=f"{name}_dropout")(hidden)
    return hidden + x, cache


def build_self_attention_mask(cache, index):
    positions = ops.ones_like(cache[:, 0, :, 0, 0], dtype="int32")
    positions = ops.cumsum(positions, axis=1) - 1
    positions = ops.expand_dims(positions, axis=1)
    return ops.cast(positions <= index, "int32")


def attend_cross(x, cross_cache, config, name):
    norm_name = f"{name}_layer_norm"
    delta = LayerNormalization(epsilon=1e-5, name=norm_name)(x)
    args = (delta, cross_cache, None, None, None, config, name)
    delta, _ = kv_attend(*args)
    delta = Dropout(config["dropout"], name=f"{name}_dropout")(delta)
    return x + delta


def feedforward(x, config, name):
    dim = config["ffn_dim"]
    dropout = config["dropout"]
    norm_name = f"{name}_layer_norm"
    delta = LayerNormalization(epsilon=1e-5, name=norm_name)(x)
    delta = build_gelu_dense(dim, f"{name}_intermediate_dense")(delta)
    delta = build_dense(x.shape[-1], f"{name}_output_dense")(delta)
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta
