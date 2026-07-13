from keras import Input, Model, ops

from paz.models.foundation.depth_anything3 import routing
from paz.models.foundation.depth_anything3.embeddings import build_view_embeddings
from paz.models.foundation.depth_anything3.backbone import build_da3_backbone
from paz.models.foundation.depth_anything3.dual_dpt import build_dual_dpt
from paz.models.foundation.depth_anything3.camera import build_camera_decoder

HIDDEN_SIZE = 384
FEATURE_SIZE = 768

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


def build_da3_small(num_views, image_shape, name="da3_small"):
    images = Input((num_views, *image_shape), name="views")
    grid = grid_shape(image_shape, PATCH_SIZE)
    num_positions = grid[0] * grid[1] + 1
    tokens = build_view_embeddings(images, PATCH_SIZE, HIDDEN_SIZE,
                                   num_positions, num_views)
    features, camera_tokens = build_da3_backbone(
        tokens, num_views, grid, HIDDEN_SIZE, 12, 6, MLP_RATIO,
        LAYER_SCALE_INIT, OUT_LAYERS, ALT_START)
    depth, depth_conf, rays, ray_conf = apply_head(features, num_views, grid,
                                                   image_shape)
    extrinsics, intrinsics = build_camera_decoder(camera_tokens[-1],
                                                  FEATURE_SIZE, image_shape)
    outputs = [depth, depth_conf, extrinsics, intrinsics, rays, ray_conf]
    return Model(images, outputs, name=name)


def apply_head(features, num_views, grid, image_shape):
    folded = [routing.fold_views_into_batch(feature) for feature in features]
    depth, depth_conf, rays, ray_conf = build_dual_dpt(folded, grid, image_shape)
    height, width = image_shape[0], image_shape[1]
    ray_height, ray_width = grid[0] * 8, grid[1] * 8
    depth = ops.reshape(depth, (-1, num_views, height, width))
    depth_conf = ops.reshape(depth_conf, (-1, num_views, height, width))
    rays = ops.reshape(rays, (-1, num_views, ray_height, ray_width, 6))
    ray_conf = ops.reshape(ray_conf, (-1, num_views, ray_height, ray_width))
    return depth, depth_conf, rays, ray_conf
