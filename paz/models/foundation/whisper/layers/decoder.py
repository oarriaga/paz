import paz
from keras import ops
from keras.layers import LayerNormalization, Dropout

from paz.models.transformers.attention import kv_attend


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
    heads, dim, drop = unpack_attention_config(config)
    args = (hidden, cache, index, hidden, mask, heads, dim, drop, name)
    hidden, cache = kv_attend(*args)
    hidden = Dropout(config["dropout"], name=f"{name}_dropout")(hidden)
    return hidden + x, cache


def unpack_attention_config(config):
    return config["num_heads"], config["hidden_dim"], config["dropout"]


def build_self_attention_mask(cache, index):
    key_positions = ops.ones_like(cache[:, 0, :, 0, 0], dtype="int32")
    key_positions = ops.cumsum(key_positions, axis=1) - 1
    query_positions = ops.reshape(index, (1, 1))
    return paz.transformers.mask.causal(query_positions, key_positions)


def attend_cross(x, cross_cache, config, name):
    norm_name = f"{name}_layer_norm"
    delta = LayerNormalization(epsilon=1e-5, name=norm_name)(x)
    heads, dim, drop = unpack_attention_config(config)
    args = (delta, cross_cache, None, None, None, heads, dim, drop, name)
    delta, _ = kv_attend(*args)
    delta = Dropout(config["dropout"], name=f"{name}_dropout")(delta)
    return x + delta


def feedforward(x, config, name):
    dim = config["ffn_dim"]
    dropout = config["dropout"]
    norm_name = f"{name}_layer_norm"
    delta = LayerNormalization(epsilon=1e-5, name=norm_name)(x)
    delta = paz.transformers.feedforward.gelu(
        delta, dim, x.shape[-1], f"{name}_intermediate_dense",
        f"{name}_output_dense")
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta
