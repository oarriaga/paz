from keras import Input, Model, ops
from keras.layers import LayerNormalization

from paz.models.foundation.depth_anything3 import routing
from paz.models.foundation.depth_anything3.embeddings import build_view_embeddings
from paz.models.foundation.depth_anything3.backbone import build_da3_backbone
from paz.models.foundation.depth_anything3.dual_dpt import build_dual_dpt
from paz.models.foundation.depth_anything3.dpt import build_dpt
from paz.models.foundation.depth_anything3.camera import build_camera_decoder
from paz.models.foundation.dinov2.embeddings import build_dinov2_embeddings
from paz.models.foundation.dinov2.encoder import build_dinov2_encoder

LAYER_NORM_EPSILON = 1e-6
MONO_OUT_LAYERS = (4, 11, 17, 23)

PATCH_SIZE = 14
MLP_RATIO = 4.0
LAYER_SCALE_INIT = 1.0
OUT_LAYERS = (5, 7, 9, 11)
ALT_START = 4


def build_da3_small_backbone(num_views, image_shape,
                             name="da3_small_backbone"):
    args = (num_views, image_shape, 384, 12, 6)
    return build_da3_backbone_model(*args, name)


def build_da3_backbone_model(num_views, image_shape, hidden_size, depth,
                             num_heads, name):
    images = Input((num_views, *image_shape), name="views")
    grid = grid_shape(image_shape, PATCH_SIZE)
    num_positions = grid[0] * grid[1] + 1
    tokens = build_view_embeddings(images, PATCH_SIZE, hidden_size,
                                   num_positions, num_views)
    features, camera_tokens = build_da3_backbone(
        tokens, num_views, grid, hidden_size, depth, num_heads, MLP_RATIO,
        LAYER_SCALE_INIT, OUT_LAYERS, ALT_START)
    return Model(images, features + camera_tokens, name=name)


def grid_shape(image_shape, patch_size):
    return image_shape[0] // patch_size, image_shape[1] // patch_size


def build_da3_mono_large(image_shape, name="da3_mono_large"):
    return build_mono(image_shape, name)


def build_da3_metric_large(image_shape, name="da3_metric_large"):
    return build_mono(image_shape, name)


def build_mono(image_shape, name):
    images = Input(image_shape, name="image")
    grid = grid_shape(image_shape, PATCH_SIZE)
    num_positions = grid[0] * grid[1] + 1
    tokens = build_dinov2_embeddings(images, PATCH_SIZE, 1024, num_positions, 0)
    _, hidden_states = build_dinov2_encoder(tokens, 1024, 24, 16, MLP_RATIO,
                                            LAYER_SCALE_INIT)
    features = collect_mono_features(hidden_states)
    depth, sky = build_dpt(features, grid, image_shape, 1024,
                           (256, 512, 1024, 1024), 256)
    return Model(images, [depth, sky], name=name)


def collect_mono_features(hidden_states):
    norm = LayerNormalization(epsilon=LAYER_NORM_EPSILON, name="norm")
    features = []
    for layer in MONO_OUT_LAYERS:
        features.append(norm(hidden_states[layer])[:, 1:])
    return features


def build_da3_small(num_views, image_shape, name="da3_small"):
    args = (384, 6, (48, 96, 192, 384), 64)
    return build_da3(num_views, image_shape, *args, name)


def build_da3_base(num_views, image_shape, name="da3_base"):
    args = (768, 12, (96, 192, 384, 768), 128)
    return build_da3(num_views, image_shape, *args, name)


def build_da3(num_views, image_shape, hidden_size, num_heads, out_channels,
              fusion_features, name):
    images = Input((num_views, *image_shape), name="views")
    grid = grid_shape(image_shape, PATCH_SIZE)
    num_positions = grid[0] * grid[1] + 1
    tokens = build_view_embeddings(images, PATCH_SIZE, hidden_size,
                                   num_positions, num_views)
    features, camera_tokens = build_da3_backbone(
        tokens, num_views, grid, hidden_size, 12, num_heads, MLP_RATIO,
        LAYER_SCALE_INIT, OUT_LAYERS, ALT_START)
    feature_dim = 2 * hidden_size
    depth, depth_conf, rays, ray_conf = apply_head(
        features, num_views, grid, image_shape, feature_dim, out_channels,
        fusion_features)
    extrinsics, intrinsics = build_camera_decoder(camera_tokens[-1],
                                                  feature_dim, image_shape)
    outputs = [depth, depth_conf, extrinsics, intrinsics, rays, ray_conf]
    return Model(images, outputs, name=name)


def apply_head(features, num_views, grid, image_shape, feature_dim,
               out_channels, fusion_features):
    folded = [routing.fold_views_into_batch(feature) for feature in features]
    depth, depth_conf, rays, ray_conf = build_dual_dpt(
        folded, grid, image_shape, feature_dim, out_channels, fusion_features)
    height, width = image_shape[0], image_shape[1]
    ray_height, ray_width = grid[0] * 8, grid[1] * 8
    depth = ops.reshape(depth, (-1, num_views, height, width))
    depth_conf = ops.reshape(depth_conf, (-1, num_views, height, width))
    rays = ops.reshape(rays, (-1, num_views, ray_height, ray_width, 6))
    ray_conf = ops.reshape(ray_conf, (-1, num_views, ray_height, ray_width))
    return depth, depth_conf, rays, ray_conf
