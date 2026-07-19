from keras import Input, Model, ops
from keras.layers import LayerNormalization

from paz.models.foundation.depth_anything3 import routing
from paz.models.foundation.depth_anything3 import embeddings
from paz.models.foundation.depth_anything3 import backbone
from paz.models.foundation.depth_anything3 import dual_dpt
from paz.models.foundation.depth_anything3 import dpt
from paz.models.foundation.depth_anything3 import camera
from paz.models.foundation.dinov2 import embeddings as dinov2_embeddings
from paz.models.foundation.dinov2 import encoder as dinov2_encoder


def DepthAnything3Small(num_views, image_shape, name="da3_small"):
    args = 384, 6, (48, 96, 192, 384), 64
    return build_da3(num_views, image_shape, *args, name)


def DepthAnything3Base(num_views, image_shape, name="da3_base"):
    args = 768, 12, (96, 192, 384, 768), 128
    return build_da3(num_views, image_shape, *args, name)


def build_da3(num_views, image_shape, hidden_size, num_heads, out_channels,
              fusion_features, name):
    images = Input((num_views, *image_shape), name="views")
    grid = grid_shape(image_shape, 14)
    tokens = embed_views(images, hidden_size, num_views, grid)
    features, cameras = encode_views(tokens, num_views, grid, hidden_size,
                                     num_heads)
    args = features, num_views, grid, image_shape, hidden_size
    head = apply_head(*args, out_channels, fusion_features)
    extrinsics, intrinsics = decode_cameras(cameras, hidden_size, image_shape)
    outputs = [head[0], head[1], extrinsics, intrinsics, head[2], head[3]]
    return Model(images, outputs, name=name)


def embed_views(images, hidden_size, num_views, grid):
    positions = grid[0] * grid[1] + 1
    args = images, 14, hidden_size, positions, num_views
    return embeddings.build_view_embeddings(*args)


def encode_views(tokens, num_views, grid, hidden_size, num_heads):
    block_args = hidden_size, num_heads, 4.0, 1.0
    args = tokens, num_views, grid, block_args, 12, (5, 7, 9, 11), 4
    return backbone.build(*args)


def apply_head(features, num_views, grid, image_shape, hidden_size,
               out_channels, fusion_features):
    folded = [routing.fold_views_into_batch(view) for view in features]
    feature_dim = 2 * hidden_size
    args = folded, grid, image_shape, feature_dim, out_channels
    outputs = dual_dpt.build(*args, fusion_features)
    return restore_views(outputs, num_views, grid, image_shape)


def restore_views(outputs, num_views, grid, image_shape):
    depth, depth_conf, rays, ray_conf = outputs
    H, W = image_shape[0], image_shape[1]
    ray_H, ray_W = grid[0] * 8, grid[1] * 8
    depth = ops.reshape(depth, (-1, num_views, H, W))
    depth_conf = ops.reshape(depth_conf, (-1, num_views, H, W))
    rays = ops.reshape(rays, (-1, num_views, ray_H, ray_W, 6))
    ray_conf = ops.reshape(ray_conf, (-1, num_views, ray_H, ray_W))
    return depth, depth_conf, rays, ray_conf


def decode_cameras(cameras, hidden_size, image_shape):
    args = cameras[-1], 2 * hidden_size, image_shape
    return camera.build_camera_decoder(*args)


def DepthAnything3MonoLarge(image_shape, name="da3_mono_large"):
    return build_mono(image_shape, name)


def DepthAnything3MetricLarge(image_shape, name="da3_metric_large"):
    return build_mono(image_shape, name)


def build_mono(image_shape, name):
    images = Input(image_shape, name="image")
    grid = grid_shape(image_shape, 14)
    positions = grid[0] * grid[1] + 1
    args = images, 14, 1024, positions, 0
    tokens = dinov2_embeddings.build(*args)
    _, hidden_states = dinov2_encoder.build(tokens, 1024, 24, 16, 4.0, 1.0)
    features = collect_mono_features(hidden_states)
    channels = 256, 512, 1024, 1024
    depth, sky = dpt.build(features, grid, image_shape, 1024, channels, 256)
    return Model(images, [depth, sky], name=name)


def collect_mono_features(hidden_states):
    norm = LayerNormalization(epsilon=1e-6, name="norm")
    features = []
    for layer in (4, 11, 17, 23):
        features.append(norm(hidden_states[layer])[:, 1:])
    return features


def build_da3_small_backbone(num_views, image_shape, name="da3_backbone"):
    images = Input((num_views, *image_shape), name="views")
    grid = grid_shape(image_shape, 14)
    tokens = embed_views(images, 384, num_views, grid)
    features, cameras = encode_views(tokens, num_views, grid, 384, 6)
    return Model(images, features + cameras, name=name)


def grid_shape(image_shape, patch_size):
    return image_shape[0] // patch_size, image_shape[1] // patch_size
