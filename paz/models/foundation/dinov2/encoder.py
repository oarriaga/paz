from paz.models.foundation.dinov2.blocks import build_dinov2_block


def build_dinov2_encoder(tokens, hidden_size, depth, num_heads, mlp_ratio,
                         layer_scale_init):
    hidden_states = []
    for index in range(depth):
        args = (hidden_size, num_heads, mlp_ratio, layer_scale_init)
        tokens = build_dinov2_block(tokens, *args, f"block_{index}")
        hidden_states.append(tokens)
    return tokens, tuple(hidden_states)
