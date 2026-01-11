import jax.numpy as jp
import paz


def compute_focal_length(y_field_of_view):
    return 1 / jp.tan(y_field_of_view / 2.0)


def compute_aspect_ratio(H, W):
    return W / H


def compute_image_sizes(y_field_of_view, aspect_ratio):
    H = 2 * jp.tan(y_field_of_view / 2.0)
    W = 2 * jp.tan(y_field_of_view / 2.0) * aspect_ratio
    return H, W


def compute_pixel_size(width_in_world_coordinates, width_in_pixels):
    return width_in_world_coordinates / width_in_pixels


def build_ray_directions(H_pixel, W_pixel, H_world, W_world):
    pixel_size = compute_pixel_size(W_world, W_pixel)
    x_offset = (jp.arange(0, W_pixel) + 0.5) * pixel_size
    y_offset = (jp.arange(0, H_pixel) + 0.5) * pixel_size
    grid = jp.meshgrid((W_world / 2) - x_offset, (H_world / 2) - y_offset)
    x_grid = jp.reshape(grid[0], [W_pixel * H_pixel, 1])
    y_grid = jp.reshape(grid[1], [W_pixel * H_pixel, 1])
    z_grid = jp.ones((W_pixel * H_pixel)).reshape([H_pixel * W_pixel, 1])
    return jp.concatenate([-x_grid, y_grid, -z_grid], axis=-1)


def build_ray_origins(H, W):
    return jp.repeat(jp.array([[0.0, 0.0, 0.0]]), repeats=H * W, axis=0)


def build_rays(size, y_FOV, world_to_camera=None):
    if world_to_camera is None:
        world_to_camera = jp.eye(4)
    H_pixel, W_pixel = size[:2]
    aspect_ratio = compute_aspect_ratio(H_pixel, W_pixel)
    H_world, W_world = compute_image_sizes(y_FOV, aspect_ratio)
    directions = build_ray_directions(H_pixel, W_pixel, H_world, W_world)
    origins = build_ray_origins(H_pixel, W_pixel)
    camera_to_world = jp.linalg.inv(world_to_camera)
    return paz.algebra.transform_rays(camera_to_world, origins, directions)


def compute_intrinsics(y_FOV, H_pixels, W_pixels):
    aspect_ratio = compute_aspect_ratio(H_pixels, W_pixels)
    y_focal_length = compute_focal_length(y_FOV)
    x_focal_length = y_focal_length / aspect_ratio
    half_W = W_pixels / 2.0
    half_H = H_pixels / 2.0
    return jp.array(
        [
            [x_focal_length * half_W, 0.0, half_W, 0.0],
            [0.0, y_focal_length * half_H, half_H, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
