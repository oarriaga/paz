import keras
from keras.layers import Add, Dense, EinsumDense, LayerNormalization

from paz.models.transformers import feedforward
from paz.models.transformers.attention import kernel
from paz.models.transformers.attention import project_query_key_value
from paz.models.transformers.attention import split_query_key_value
from paz.models.transformers.attention import compute_attention
from paz.models.transformers.attention import merge_attention_heads


def build(tokens, hidden_size, num_heads, MLP_ratio, scale_init, name):
    attended = apply_attention(tokens, hidden_size, num_heads, name)
    tokens = add_residual(tokens, attended, hidden_size, scale_init, name, 1)
    forwarded = apply_feedforward(tokens, hidden_size, MLP_ratio, name)
    return add_residual(tokens, forwarded, hidden_size, scale_init, name, 2)


def add_residual(tokens, branch, hidden_size, scale_init, name, index):
    scale_name = f"{name}_ls{index}"
    scaled = apply_layer_scale(branch, hidden_size, scale_init, scale_name)
    return Add(name=f"{name}_add{index}")([tokens, scaled])


def apply_attention(tokens, hidden_size, num_heads, name):
    normed = normalize(tokens, f"{name}_norm1")
    fused = project_query_key_value(normed, hidden_size, True, name)
    head_dim = hidden_size // num_heads
    query, key, value = split_query_key_value(fused, num_heads, head_dim)
    context = compute_attention(query, key, value)
    merged = merge_attention_heads(context)
    return project_output(merged, hidden_size, f"{name}_proj")


def apply_feedforward(tokens, hidden_size, MLP_ratio, name):
    normed = normalize(tokens, f"{name}_norm2")
    inner_size = int(hidden_size * MLP_ratio)
    names = f"{name}_mlp_fc1", f"{name}_mlp_fc2"
    return feedforward.gelu(normed, inner_size, hidden_size, *names)


def project_output(tokens, hidden_size, name):
    kwargs = dict(use_bias=True, kernel_initializer=kernel(), name=name)
    return Dense(hidden_size, **kwargs)(tokens)


def apply_layer_scale(tokens, hidden_size, scale_init, name):
    initializer = keras.initializers.Constant(scale_init)
    kwargs = dict(output_shape=(hidden_size,), bias_axes=None, name=name)
    kwargs["kernel_initializer"] = initializer
    return EinsumDense("...d,d->...d", **kwargs)(tokens)


def normalize(tokens, name):
    return LayerNormalization(epsilon=1e-6, name=name)(tokens)
