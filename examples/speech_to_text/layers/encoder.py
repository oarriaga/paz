from keras.layers import LayerNormalization, Dropout

from examples.speech_to_text.layers.utils import build_gelu_dense, build_dense
from examples.speech_to_text.layers.attention import attend


def encoder_block(x, num_heads, dim, name, dropout, epsilon):
    key_dim = int(x.shape[-1] // num_heads)
    attn_name = f"{name}_self_attention_layer"
    x = self_attend(x, num_heads, key_dim, dropout, epsilon, attn_name)
    ff_name = f"{name}_feedforward"
    return encoder_dense(x, dim, dropout, epsilon, ff_name)


def self_attend(x, num_heads, key_dim, dropout, epsilon, name):
    norm_name = f"{name}_norm"
    norm = LayerNormalization(epsilon=epsilon, name=norm_name)
    normalized = norm(x)
    delta = attend(normalized, normalized, num_heads, key_dim, dropout, name)
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta


def encoder_dense(x, dim, dropout, epsilon, name):
    norm_name = f"{name}_layer_norm"
    delta = LayerNormalization(epsilon=epsilon, name=norm_name)(x)
    delta = build_gelu_dense(dim, f"{name}_intermediate_dense")(delta)
    delta = build_dense(x.shape[-1], f"{name}_output_dense")(delta)
    delta = Dropout(dropout, name=f"{name}_dropout")(delta)
    return x + delta
