from paz.models.foundation.dinov2 import blocks


def build(tokens, hidden_size, depth, num_heads, MLP_ratio, scale_init):
    hidden_states = []
    for arg in range(depth):
        args = (hidden_size, num_heads, MLP_ratio, scale_init)
        tokens = blocks.build(tokens, *args, f"block_{arg}")
        hidden_states.append(tokens)
    return tokens, tuple(hidden_states)
