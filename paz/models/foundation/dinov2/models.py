from keras import Input, Model
from keras.layers import LayerNormalization, Reshape

from paz.models.foundation.dinov2 import embeddings, encoder
from paz.models.transformers import windowing


def build_dinov2(image_shape, patch_size, hidden_size, depth, num_heads,
                 MLP_ratio, num_registers, name):
    images = Input(image_shape, name="pixels")
    positions = count_positions(image_shape, patch_size)
    args = images, patch_size, hidden_size, positions, num_registers
    tokens = embeddings.build(*args)
    args = hidden_size, depth, num_heads, MLP_ratio, 1e-5
    tokens, _ = encoder.build(tokens, *args)
    normalized = normalize_tokens(tokens)
    class_token = normalized[:, 0]
    patch_tokens = normalized[:, 1 + num_registers:]
    return Model(images, (class_token, patch_tokens), name=name)


def build_dinov2_features(image_shape, patch_size, hidden_size, depth,
                          num_heads, MLP_ratio, num_registers, out_layers,
                          name):
    images = Input(image_shape, name="pixels")
    positions = count_positions(image_shape, patch_size)
    args = images, patch_size, hidden_size, positions, num_registers
    tokens = embeddings.build(*args)
    args = hidden_size, depth, num_heads, MLP_ratio, 1e-5
    _, hidden_states = encoder.build(tokens, *args)
    grid = grid_shape(image_shape, patch_size)
    maps = select_feature_maps(hidden_states, out_layers, num_registers,
                               grid, hidden_size)
    return Model(images, maps, name=name)


def build_dinov2_windowed_features(image_shape, patch_size, hidden_size,
                                   depth, num_heads, MLP_ratio, num_windows,
                                   global_layers, out_layers, name):
    """Windowed feature backbone, as used by the RF-DETR detectors.

    Attention stays inside ``num_windows`` squared windows except in
    ``global_layers``. Returns one channels-last map per tapped layer, each
    normalized by the same final layer norm.
    """
    images = Input(image_shape, name="pixels")
    grid = grid_shape(image_shape, patch_size)
    args = images, patch_size, hidden_size, grid, num_windows
    tokens = embeddings.build_windowed(*args)
    args = hidden_size, depth, num_heads, MLP_ratio, 1e-5, num_windows
    _, hidden_states = encoder.build_windowed(tokens, *args, global_layers)
    maps = select_windowed_maps(hidden_states, out_layers, grid, num_windows)
    return Model(images, maps, name=name)


def select_windowed_maps(hidden_states, out_layers, grid, num_windows):
    normalize = LayerNormalization(epsilon=1e-6, name="norm")
    maps = []
    for arg in out_layers:
        patch_tokens = normalize(hidden_states[arg])[:, 1:]
        maps.append(windowing.unpartition(patch_tokens, grid, num_windows))
    return maps


def select_feature_maps(hidden_states, out_layers, skip, grid, hidden_size):
    height, width = grid
    maps = []
    for arg in out_layers:
        patch_tokens = hidden_states[arg][:, 1 + skip:]
        maps.append(Reshape((height, width, hidden_size))(patch_tokens))
    return maps


def normalize_tokens(tokens):
    return LayerNormalization(epsilon=1e-6, name="norm")(tokens)


def count_positions(image_shape, patch_size):
    height, width = grid_shape(image_shape, patch_size)
    return height * width + 1


def grid_shape(image_shape, patch_size):
    return image_shape[0] // patch_size, image_shape[1] // patch_size


def DINOv2Small(image_shape=(518, 518, 3), num_registers=0,
                name="dinov2_small"):
    args = image_shape, 14, 384, 12, 6, 4.0
    return build_dinov2(*args, num_registers, name)


def DINOv2Base(image_shape=(518, 518, 3), num_registers=0,
               name="dinov2_base"):
    args = image_shape, 14, 768, 12, 12, 4.0
    return build_dinov2(*args, num_registers, name)


def DINOv2Large(image_shape=(518, 518, 3), num_registers=0,
                name="dinov2_large"):
    args = image_shape, 14, 1024, 24, 16, 4.0
    return build_dinov2(*args, num_registers, name)


def DINOv2SmallFeatures(image_shape=(518, 518, 3), out_layers=(5, 7, 9, 11),
                        num_registers=0, name="dinov2_small_features"):
    args = image_shape, 14, 384, 12, 6, 4.0
    return build_dinov2_features(*args, num_registers, out_layers, name)


def DINOv2SmallWindowedFeatures(image_shape, patch_size, num_windows,
                                global_layers, out_layers,
                                name="dinov2_small_windowed_features"):
    args = image_shape, patch_size, 384, 12, 6, 4.0, num_windows
    args = args + (global_layers, out_layers, name)
    return build_dinov2_windowed_features(*args)


def DINOv2BaseFeatures(image_shape=(518, 518, 3), out_layers=(5, 7, 9, 11),
                       num_registers=0, name="dinov2_base_features"):
    args = image_shape, 14, 768, 12, 12, 4.0
    return build_dinov2_features(*args, num_registers, out_layers, name)


def DINOv2LargeFeatures(image_shape=(518, 518, 3), out_layers=(5, 7, 9, 11),
                        num_registers=0, name="dinov2_large_features"):
    args = image_shape, 14, 1024, 24, 16, 4.0
    return build_dinov2_features(*args, num_registers, out_layers, name)
