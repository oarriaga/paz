import keras
from keras.layers import Add, Dense, EinsumDense, LayerNormalization

from paz.models.transformers import attention, feedforward

LAYER_NORM_EPSILON = 1e-6


def build_dinov2_block(tokens, hidden_size, num_heads, mlp_ratio,
                       layer_scale_init, name):
    attended = apply_attention(tokens, hidden_size, num_heads, name)
    scaled = apply_layer_scale(attended, hidden_size, layer_scale_init,
                               f"{name}_ls1")
    tokens = Add(name=f"{name}_add1")([tokens, scaled])
    forwarded = apply_feedforward(tokens, hidden_size, mlp_ratio, name)
    scaled = apply_layer_scale(forwarded, hidden_size, layer_scale_init,
                               f"{name}_ls2")
    return Add(name=f"{name}_add2")([tokens, scaled])


def apply_attention(tokens, hidden_size, num_heads, name):
    normed = normalize(tokens, f"{name}_norm1")
    fused = attention.project_query_key_value(normed, hidden_size, True, name)
    head_dim = hidden_size // num_heads
    query, key, value = attention.split_query_key_value(fused, num_heads,
                                                         head_dim)
    context = attention.compute_attention(query, key, value)
    merged = attention.merge_attention_heads(context)
    return project_output(merged, hidden_size, f"{name}_proj")


def apply_feedforward(tokens, hidden_size, mlp_ratio, name):
    normed = normalize(tokens, f"{name}_norm2")
    inner_size = int(hidden_size * mlp_ratio)
    return feedforward.gelu(normed, inner_size, hidden_size,
                            f"{name}_mlp_fc1", f"{name}_mlp_fc2")


def project_output(tokens, hidden_size, name):
    layer = Dense(hidden_size, use_bias=True,
                  kernel_initializer=attention.kernel(), name=name)
    return layer(tokens)


def apply_layer_scale(tokens, hidden_size, init, name):
    initializer = keras.initializers.Constant(init)
    layer = EinsumDense("...d,d->...d", output_shape=(hidden_size,),
                        bias_axes=None, kernel_initializer=initializer,
                        name=name)
    return layer(tokens)


def normalize(tokens, name):
    return LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name)(tokens)
