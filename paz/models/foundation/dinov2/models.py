from keras import Input, Model
from keras.layers import LayerNormalization, Reshape

from paz.models.foundation.dinov2.embeddings import build_dinov2_embeddings
from paz.models.foundation.dinov2.encoder import build_dinov2_encoder

PATCH_SIZE = 14
MLP_RATIO = 4.0
LAYER_SCALE_INIT = 1e-5
LAYER_NORM_EPSILON = 1e-6
DEFAULT_IMAGE_SHAPE = (518, 518, 3)
DEFAULT_OUT_LAYERS = (5, 7, 9, 11)


def build_dinov2(image_shape, patch_size, hidden_size, depth, num_heads,
                 mlp_ratio, num_register_tokens, name):
    images = Input(image_shape, name="pixels")
    positions = count_positions(image_shape, patch_size)
    tokens = build_dinov2_embeddings(images, patch_size, hidden_size,
                                     positions, num_register_tokens)
    args = (hidden_size, depth, num_heads, mlp_ratio, LAYER_SCALE_INIT)
    tokens, _ = build_dinov2_encoder(tokens, *args)
    normalized = normalize_tokens(tokens)
    class_token = normalized[:, 0]
    patch_tokens = normalized[:, 1 + num_register_tokens:]
    return Model(images, (class_token, patch_tokens), name=name)


def build_dinov2_features(image_shape, patch_size, hidden_size, depth,
                          num_heads, mlp_ratio, num_register_tokens,
                          out_layers, name):
    images = Input(image_shape, name="pixels")
    positions = count_positions(image_shape, patch_size)
    tokens = build_dinov2_embeddings(images, patch_size, hidden_size,
                                     positions, num_register_tokens)
    args = (hidden_size, depth, num_heads, mlp_ratio, LAYER_SCALE_INIT)
    _, hidden_states = build_dinov2_encoder(tokens, *args)
    grid = grid_shape(image_shape, patch_size)
    maps = select_feature_maps(hidden_states, out_layers,
                               num_register_tokens, grid, hidden_size)
    return Model(images, maps, name=name)


def select_feature_maps(hidden_states, out_layers, skip, grid, hidden_size):
    height, width = grid
    maps = []
    for index in out_layers:
        patch_tokens = hidden_states[index][:, 1 + skip:]
        maps.append(Reshape((height, width, hidden_size))(patch_tokens))
    return maps


def normalize_tokens(tokens):
    return LayerNormalization(epsilon=LAYER_NORM_EPSILON, name="norm")(tokens)


def count_positions(image_shape, patch_size):
    height, width = grid_shape(image_shape, patch_size)
    return height * width + 1


def grid_shape(image_shape, patch_size):
    return image_shape[0] // patch_size, image_shape[1] // patch_size


def DINOv2Small(image_shape=DEFAULT_IMAGE_SHAPE, num_register_tokens=0,
                name="dinov2_small"):
    args = (image_shape, PATCH_SIZE, 384, 12, 6, MLP_RATIO)
    return build_dinov2(*args, num_register_tokens, name)


def DINOv2Base(image_shape=DEFAULT_IMAGE_SHAPE, num_register_tokens=0,
               name="dinov2_base"):
    args = (image_shape, PATCH_SIZE, 768, 12, 12, MLP_RATIO)
    return build_dinov2(*args, num_register_tokens, name)


def DINOv2Large(image_shape=DEFAULT_IMAGE_SHAPE, num_register_tokens=0,
                name="dinov2_large"):
    args = (image_shape, PATCH_SIZE, 1024, 24, 16, MLP_RATIO)
    return build_dinov2(*args, num_register_tokens, name)


def DINOv2SmallFeatures(image_shape=DEFAULT_IMAGE_SHAPE,
                        out_layers=DEFAULT_OUT_LAYERS, num_register_tokens=0,
                        name="dinov2_small_features"):
    args = (image_shape, PATCH_SIZE, 384, 12, 6, MLP_RATIO)
    return build_dinov2_features(*args, num_register_tokens, out_layers, name)


def DINOv2BaseFeatures(image_shape=DEFAULT_IMAGE_SHAPE,
                       out_layers=DEFAULT_OUT_LAYERS, num_register_tokens=0,
                       name="dinov2_base_features"):
    args = (image_shape, PATCH_SIZE, 768, 12, 12, MLP_RATIO)
    return build_dinov2_features(*args, num_register_tokens, out_layers, name)


def DINOv2LargeFeatures(image_shape=DEFAULT_IMAGE_SHAPE,
                        out_layers=DEFAULT_OUT_LAYERS, num_register_tokens=0,
                        name="dinov2_large_features"):
    args = (image_shape, PATCH_SIZE, 1024, 24, 16, MLP_RATIO)
    return build_dinov2_features(*args, num_register_tokens, out_layers, name)
