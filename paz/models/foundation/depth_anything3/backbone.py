from keras import ops
from keras.layers import LayerNormalization

from paz.models.foundation.dinov2 import blocks as dinov2_blocks
from paz.models.foundation.depth_anything3 import routing
from paz.models.foundation.depth_anything3 import blocks
from paz.models.foundation.depth_anything3 import embeddings


def build(tokens, num_views, grid, args, depth, out_layers, alternation_start):
    positions = build_grid_positions(num_views, grid)
    context = num_views, positions, args, alternation_start
    collected = encode(tokens, context, depth, out_layers)
    return finalize_features(collected, args[0])


def encode(tokens, context, depth, out_layers):
    num_views, positions, args, alternation_start = context
    local_tokens, collected = tokens, []
    for arg in range(depth):
        tokens = apply_block(tokens, arg, context)
        if not is_global(arg, alternation_start):
            local_tokens = tokens
        collect_feature(collected, local_tokens, tokens, arg, out_layers)
    return collected


def apply_block(tokens, arg, context):
    num_views, positions, args, alternation_start = context
    if arg == alternation_start:
        tokens = embeddings.insert_camera_tokens(tokens, args[0])
    if is_global(arg, alternation_start):
        return apply_global(tokens, arg, context)
    return apply_local(tokens, arg, context)


def apply_local(tokens, arg, context):
    num_views, positions, args, alternation_start = context
    folded = routing.fold_views_into_batch(tokens)
    name = f"block_{arg}"
    if arg < alternation_start:
        folded = dinov2_blocks.build(folded, *args, name)
    else:
        folded = blocks.build(folded, positions[0], args, name)
    return routing.restore_view_dimension(folded, num_views)


def apply_global(tokens, arg, context):
    num_views, positions, args, alternation_start = context
    merged = routing.merge_views_into_sequence(tokens)
    merged = blocks.build(merged, positions[1], args, f"block_{arg}")
    return routing.split_sequence_into_views(merged, num_views)


def is_global(arg, alternation_start):
    reached_alternation = arg >= alternation_start
    is_odd = arg % 2 == 1
    return reached_alternation and is_odd


def collect_feature(collected, local_tokens, tokens, arg, out_layers):
    if arg in out_layers:
        feature = ops.concatenate([local_tokens, tokens], axis=-1)
        collected.append(feature)


def finalize_features(collected, hidden_size):
    norm = LayerNormalization(epsilon=1e-6, name="norm")
    features, camera_tokens = [], []
    for feature in collected:
        camera_tokens.append(feature[:, :, 0])
        normed = normalize_global_half(feature, hidden_size, norm)
        features.append(normed[:, :, 1:, :])
    return features, camera_tokens


def normalize_global_half(feature, hidden_size, norm):
    local_half = feature[..., :hidden_size]
    global_half = norm(feature[..., hidden_size:])
    return ops.concatenate([local_half, global_half], axis=-1)


def build_grid_positions(num_views, grid):
    local = routing.build_local_positions(*grid)
    cross_view = routing.build_global_positions(num_views, *grid)
    return local, cross_view
