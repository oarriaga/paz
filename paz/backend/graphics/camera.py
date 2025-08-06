import jax.numpy as jp

import paz


def compute_camera_intrinsics(y_FOV, H, W):
    aspect_ratio = W / H
    y_focal_length = 1 / (jp.tan(y_FOV / 2.0))
    x_focal_length = y_focal_length / aspect_ratio

    x_translation = W / 2.0
    y_translation = H / 2.0
    return jp.array(
        [
            [x_focal_length * (W / 2.0), 0.0, x_translation, 0.0],
            [0.0, y_focal_length * (H / 2.0), y_translation, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


def compute_focal_length(y_field_of_view):
    focal_length = 1 / jp.tan(y_field_of_view / 2.0)
    return focal_length


def compute_half_view(y_field_of_view, aspect_ratio):
    # TODO check why aspect_ratio is needed. Report back bug.
    half_view = jp.tan(y_field_of_view / 2.0) * aspect_ratio
    return half_view


def compute_aspect_ratio(height, width):
    aspect_ratio = width / height
    return aspect_ratio


def compute_half_W(aspect_ratio, half_view):
    if aspect_ratio >= 1.0:
        half_W = half_view
    else:
        half_W = half_view * aspect_ratio
    return half_W


def compute_half_H(aspect_ratio, half_view):
    if aspect_ratio >= 1.0:
        half_H = half_view / aspect_ratio
    else:
        half_H = half_view
    return half_H


def compute_pixel_size(half_W, width_in_pixels):
    pixel_size = half_W * 2.0 / width_in_pixels
    return pixel_size


def build_ray_directions(H, W, pixel_size, half_W, half_H):
    x_offset = (jp.arange(0, W) + 0.5) * pixel_size
    y_offset = (jp.arange(0, H) + 0.5) * pixel_size
    x = half_W - x_offset
    y = half_H - y_offset
    x_grid, y_grid = jp.meshgrid(x, y)
    x_grid = jp.reshape(x_grid, [W * H, 1])
    y_grid = jp.reshape(y_grid, [W * H, 1])
    z_grid = jp.repeat(-1.0, W * H)
    z_grid = jp.reshape(z_grid, [W * H, 1])
    ones = jp.ones([W * H, 1])
    ray_directions = [x_grid, y_grid, z_grid, ones]
    ray_directions = jp.concatenate(ray_directions, axis=1)
    return ray_directions


def build_ray_origins(H, W):
    ray_origins = jp.array([[0.0, 0.0, 0.0, 1.0]])
    ray_origins = jp.repeat(ray_origins, repeats=H * W, axis=0)
    return ray_origins


def transform_rays(world_to_camera, ray_origins, ray_directions):
    # (4 x 4) x (4 x N) -> (N, 4)
    ray_origins = jp.matmul(world_to_camera, ray_origins.T).T
    # (4 x 4) x (4 x N) -> (N, 4)
    ray_directions = jp.matmul(world_to_camera, ray_directions.T).T
    # (N x 4) - (N x 4) -> (N, 4)
    ray_directions = ray_directions[:, :3] - ray_origins[:, :3]
    # ray_directions = paz.algebra.normalize(ray_directions)
    ray_directions = paz.algebra.normalize(ray_directions)
    return ray_origins[:, :3], ray_directions[:, :3]


def build_rays(size, y_FOV, transform=jp.eye(4)):
    H, W = size[:2]
    aspect_ratio = compute_aspect_ratio(H, W)
    half_view = compute_half_view(y_FOV, aspect_ratio)
    half_W = compute_half_W(aspect_ratio, half_view)
    half_H = compute_half_H(aspect_ratio, half_view)
    pixel_size = compute_pixel_size(half_W, W)
    directions = build_ray_directions(H, W, pixel_size, half_W, half_H)
    origins = build_ray_origins(H, W)
    world_to_camera = jp.linalg.inv(transform)
    return transform_rays(world_to_camera, origins, directions)
