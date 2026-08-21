from paz.models.foundation.dinov2 import blocks


def build(tokens, hidden_size, depth, num_heads, MLP_ratio, scale_init):
    hidden_states = []
    for arg in range(depth):
        args = (hidden_size, num_heads, MLP_ratio, scale_init)
        tokens = blocks.build(tokens, *args, f"block_{arg}")
        hidden_states.append(tokens)
    return tokens, tuple(hidden_states)


def build_windowed(tokens, hidden_size, depth, num_heads, MLP_ratio,
                   scale_init, num_windows, global_layers):
    """Runs windowed blocks, letting ``global_layers`` attend image wide."""
    hidden_states = []
    for arg in range(depth):
        args = (hidden_size, num_heads, MLP_ratio, scale_init)
        if arg in global_layers:
            build_block = blocks.build_global
            args = args + (num_windows,)
        else:
            build_block = blocks.build
        tokens = build_block(tokens, *args, f"block_{arg}")
        hidden_states.append(tokens)
    return tokens, tuple(hidden_states)
