"""Florence-2 BART text encoder (post-norm layers, GELU feedforward)."""
from keras.layers import LayerNormalization

from paz.models.transformers import feedforward
from paz.models.transformers.attention import masked_attend

NORM_EPSILON = 1e-5


def build_encoder(x, mask, num_layers, num_heads, ffn_dim):
    for layer in range(num_layers):
        x = encoder_block(x, mask, num_heads, ffn_dim,
                          f"encoder_layer_{layer}")
    return x


def encoder_block(x, mask, num_heads, ffn_dim, name):
    dim = x.shape[-1]
    attention_name = f"{name}_self_attention"
    delta = masked_attend(x, x, mask, num_heads, dim // num_heads, 0.0,
                          attention_name)
    x = build_norm(f"{attention_name}_norm")(x + delta)
    ffn_args = (x, ffn_dim, dim, f"{name}_fc1", f"{name}_fc2")
    delta = feedforward.gelu(*ffn_args)
    return build_norm(f"{name}_final_norm")(x + delta)


def build_norm(name):
    return LayerNormalization(epsilon=NORM_EPSILON, name=name)
