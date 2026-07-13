from keras.layers import Add

from paz.models.transformers import attention
from paz.models.transformers.embeddings import rotary
from paz.models.foundation.dinov2.blocks import apply_feedforward
from paz.models.foundation.dinov2.blocks import apply_layer_scale
from paz.models.foundation.dinov2.blocks import normalize
from paz.models.foundation.dinov2.blocks import project_output

QK_NORM_EPSILON = 1e-5


def build_da3_block(tokens, positions, hidden_size, num_heads, mlp_ratio,
                    layer_scale_init, name):
    attended = apply_da3_attention(tokens, positions, hidden_size, num_heads,
                                   name)
    scaled = apply_layer_scale(attended, hidden_size, layer_scale_init,
                               f"{name}_ls1")
    tokens = Add(name=f"{name}_add1")([tokens, scaled])
    forwarded = apply_feedforward(tokens, hidden_size, mlp_ratio, name)
    scaled = apply_layer_scale(forwarded, hidden_size, layer_scale_init,
                               f"{name}_ls2")
    return Add(name=f"{name}_add2")([tokens, scaled])


def apply_da3_attention(tokens, positions, hidden_size, num_heads, name):
    normed = normalize(tokens, f"{name}_norm1")
    fused = attention.project_query_key_value(normed, hidden_size, True, name)
    head_dim = hidden_size // num_heads
    query, key, value = attention.split_query_key_value(fused, num_heads,
                                                        head_dim)
    query, key = attention.normalize_query_key(query, key, QK_NORM_EPSILON, name)
    query = rotary.apply_2d(query, positions)
    key = rotary.apply_2d(key, positions)
    context = attention.compute_attention(query, key, value)
    merged = attention.merge_attention_heads(context)
    return project_output(merged, hidden_size, f"{name}_proj")
