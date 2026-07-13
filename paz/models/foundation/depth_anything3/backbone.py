from keras import ops
from keras.layers import LayerNormalization

from paz.models.foundation.dinov2.blocks import build_dinov2_block
from paz.models.foundation.depth_anything3 import routing
from paz.models.foundation.depth_anything3.blocks import build_da3_block
from paz.models.foundation.depth_anything3.embeddings import insert_camera_tokens

LAYER_NORM_EPSILON = 1e-6


def build_da3_backbone(tokens, num_views, grid, hidden_size, depth, num_heads,
                       mlp_ratio, layer_scale_init, out_layers, alt_start):
    positions = build_positions(num_views, grid)
    collected = run_da3_encoder(tokens, num_views, positions, hidden_size,
                                depth, num_heads, mlp_ratio, layer_scale_init,
                                out_layers, alt_start)
    return finalize_features(collected, hidden_size)


def run_da3_encoder(tokens, num_views, positions, hidden_size, depth,
                    num_heads, mlp_ratio, layer_scale_init, out_layers,
                    alt_start):
    local_positions, global_positions = positions
    local_tokens = tokens
    collected = []
    for index in range(depth):
        if index == alt_start:
            tokens = insert_camera_tokens(tokens, hidden_size)
        args = (hidden_size, num_heads, mlp_ratio, layer_scale_init, index)
        if is_global_block(index, alt_start):
            tokens = apply_global_block(tokens, num_views, global_positions, *args)
        else:
            tokens = apply_local_block(tokens, num_views, local_positions,
                                       *args, alt_start)
            local_tokens = tokens
        if index in out_layers:
            collected.append(ops.concatenate([local_tokens, tokens], axis=-1))
    return collected


def apply_local_block(tokens, num_views, positions, hidden_size, num_heads,
                      mlp_ratio, layer_scale_init, index, alt_start):
    folded = routing.fold_views_into_batch(tokens)
    name = f"block_{index}"
    if index < alt_start:
        folded = build_dinov2_block(folded, hidden_size, num_heads, mlp_ratio,
                                    layer_scale_init, name)
    else:
        folded = build_da3_block(folded, positions, hidden_size, num_heads,
                                 mlp_ratio, layer_scale_init, name)
    return routing.restore_view_dimension(folded, num_views)


def apply_global_block(tokens, num_views, positions, hidden_size, num_heads,
                       mlp_ratio, layer_scale_init, index):
    merged = routing.merge_views_into_sequence(tokens)
    merged = build_da3_block(merged, positions, hidden_size, num_heads,
                             mlp_ratio, layer_scale_init, f"block_{index}")
    return routing.split_sequence_into_views(merged, num_views)


def is_global_block(index, alt_start):
    return index >= alt_start and index % 2 == 1


def finalize_features(collected, hidden_size):
    norm = LayerNormalization(epsilon=LAYER_NORM_EPSILON, name="norm")
    features, camera_tokens = [], []
    for feature in collected:
        camera_tokens.append(feature[:, :, 0])
        local_half = feature[..., :hidden_size]
        global_half = norm(feature[..., hidden_size:])
        normed = ops.concatenate([local_half, global_half], axis=-1)
        features.append(normed[:, :, 1:, :])
    return features, camera_tokens


def build_positions(num_views, grid):
    local = routing.build_local_positions(*grid)
    global_positions = routing.build_global_positions(num_views, *grid)
    return local, global_positions
