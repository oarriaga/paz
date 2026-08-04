from paz.models.transformers.embeddings import rotary
from paz.models.transformers.attention import project_query_key_value
from paz.models.transformers.attention import split_query_key_value
from paz.models.transformers.attention import normalize_query_key
from paz.models.transformers.attention import compute_attention
from paz.models.transformers.attention import merge_attention_heads
from paz.models.foundation.dinov2.blocks import add_residual
from paz.models.foundation.dinov2.blocks import apply_feedforward
from paz.models.foundation.dinov2.blocks import normalize
from paz.models.foundation.dinov2.blocks import project_output


def build(tokens, positions, args, name):
    hidden_size, num_heads, MLP_ratio, scale_init = args
    attended = apply_attention(tokens, positions, hidden_size, num_heads, name)
    tokens = add_residual(tokens, attended, hidden_size, scale_init, name, 1)
    forwarded = apply_feedforward(tokens, hidden_size, MLP_ratio, name)
    return add_residual(tokens, forwarded, hidden_size, scale_init, name, 2)


def apply_attention(tokens, positions, hidden_size, num_heads, name):
    normed = normalize(tokens, f"{name}_norm1")
    fused = project_query_key_value(normed, hidden_size, True, name)
    head_dim = hidden_size // num_heads
    query, key, value = split_query_key_value(fused, num_heads, head_dim)
    query, key = normalize_query_key(query, key, 1e-5, name)
    query = rotary.apply_2D(query, positions)
    key = rotary.apply_2D(key, positions)
    context = compute_attention(query, key, value)
    merged = merge_attention_heads(context)
    return project_output(merged, hidden_size, f"{name}_proj")
