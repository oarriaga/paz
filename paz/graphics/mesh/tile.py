from functools import partial

import jax
import jax.numpy as jp

import paz

from .render import render, render_depth


def tile_render(tile_shape, y_FOV, H, W, world_to_camera, meshes, mask, lights, chunk_size=1024):
    H_tiles, W_tiles = tile_shape
    assert_exact_tile_side(H, H_tiles)
    assert_exact_tile_side(W, W_tiles)
    args = (H, W, *tile_shape, y_FOV, world_to_camera, meshes, mask, lights, chunk_size)
    _render = partial(render_tile, *args)
    tile_coordinates = make_tile_coordinates(H_tiles, W_tiles)
    image, depth = jax.lax.scan(_render, None, tile_coordinates)[1]
    image = assemble(H, W, H_tiles, W_tiles, image)
    depth = assemble(H, W, H_tiles, W_tiles, depth)[..., 0]
    return image, depth


def tile_render_depth(tile_shape, y_FOV, H, W, world_to_camera, meshes, mask, lights, chunk_size=1024):
    H_tiles, W_tiles = tile_shape
    assert_exact_tile_side(H, H_tiles)
    assert_exact_tile_side(W, W_tiles)
    args = (H, W, *tile_shape, y_FOV, world_to_camera, meshes, mask, lights, chunk_size)
    _render = partial(render_depth_tile, *args)
    tile_coordinates = make_tile_coordinates(H_tiles, W_tiles)
    depths = jax.lax.scan(_render, None, tile_coordinates)[1]
    return assemble(H, W, H_tiles, W_tiles, depths)[..., 0]


def tile_render_masks(tile_shape, y_FOV, H, W, world_to_camera, meshes, lights, min_depth, max_depth, chunk_size=1024):
    num_meshes = len(meshes.vertices)
    masks = []
    for arg in range(num_meshes):
        mask = jp.zeros(num_meshes, dtype=bool).at[arg].set(True)
        args = tile_shape, y_FOV, H, W, world_to_camera, meshes, mask, lights
        depth = tile_render_depth(*args, chunk_size)
        soft = paz.depth.to_soft_mask(depth, min_depth, max_depth)
        masks.append(jp.expand_dims(soft, axis=-1))
    return jp.stack(masks)


def assert_exact_tile_side(image_size, tile_size):
    if (image_size / tile_size) % 1 != 0:
        raise ValueError("tile size must divide image size without a residual")


def render_tile(H, W, H_tiles, W_tiles, y_FOV, world_to_camera, meshes, mask, lights, chunk_size, carry, tile_arg):  # fmt: skip
    camera_to_world = jp.linalg.inv(world_to_camera)
    tile_shape = (H_tiles, W_tiles)
    rays = build_tile_rays(H, W, *tile_shape, y_FOV, camera_to_world, tile_arg)
    tile_shape = (H // H_tiles, W // W_tiles)
    tile = render(tile_shape, world_to_camera, rays, meshes, mask, lights, chunk_size)
    return carry, tile


def render_depth_tile(H, W, H_tiles, W_tiles, y_FOV, world_to_camera, meshes, mask, lights, chunk_size, carry, tile_arg):  # fmt: skip
    camera_to_world = jp.linalg.inv(world_to_camera)
    tile_shape = (H_tiles, W_tiles)
    rays = build_tile_rays(H, W, *tile_shape, y_FOV, camera_to_world, tile_arg)
    tile_shape = (H // H_tiles, W // W_tiles)
    depth = render_depth(tile_shape, world_to_camera, rays, meshes, mask, lights, chunk_size)
    return carry, depth


def build_tile_rays(H, W, H_tiles, W_tiles, y_FOV, camera_to_world, tile_arg):
    aspect_ratio = paz.graphics.camera.compute_aspect_ratio(H, W)
    H_world, W_world = paz.graphics.camera.compute_image_sizes(
        y_FOV, aspect_ratio
    )
    half_W = W_world / 2
    half_H = H_world / 2
    pixel_size = paz.graphics.camera.compute_pixel_size(W_world, W)
    tile_H = H // H_tiles
    tile_W = W // W_tiles
    args = (tile_H, tile_W, pixel_size, half_W, half_H, tile_arg)
    ray_targets = make_ray_targets(*args)
    ray_origins = make_ray_origins(tile_H, tile_W)
    return transform_tile_rays(camera_to_world, ray_origins, ray_targets)


def make_ray_targets(tile_H, tile_W, pixel_size, half_W, half_H, tile_arg):
    W_tile_arg, H_tile_arg = tile_arg
    W_start = tile_W * W_tile_arg
    H_start = tile_H * H_tile_arg
    x_offset = (jp.arange(tile_W) + W_start + 0.5) * pixel_size
    y_offset = (jp.arange(tile_H) + H_start + 0.5) * pixel_size
    x = x_offset - half_W
    y = half_H - y_offset
    x_grid, y_grid = jp.meshgrid(x, y)
    num_pixels = tile_W * tile_H
    x_grid = jp.reshape(x_grid, [num_pixels, 1])
    y_grid = jp.reshape(y_grid, [num_pixels, 1])
    z_grid = jp.reshape(jp.repeat(-1.0, num_pixels), [num_pixels, 1])
    ones = jp.ones([num_pixels, 1])
    ray_targets = jp.concatenate([x_grid, y_grid, z_grid, ones], axis=1)
    return ray_targets


def make_ray_origins(tile_H, tile_W):
    num_pixels = tile_H * tile_W
    ray_origins = jp.array([[0.0, 0.0, 0.0, 1.0]])
    ray_origins = jp.repeat(ray_origins, repeats=num_pixels, axis=0)
    return ray_origins


def transform_tile_rays(camera_to_world, ray_origins, ray_targets):
    origins = jp.matmul(camera_to_world, ray_origins.T).T
    targets = jp.matmul(camera_to_world, ray_targets.T).T
    directions = targets[:, :3] - origins[:, :3]
    directions = paz.algebra.normalize(directions)
    return origins[:, :3], directions


def make_tile_coordinates(H_tiles, W_tiles):
    x_args = jp.arange(W_tiles)
    y_args = jp.arange(H_tiles)
    x_grid, y_grid = jp.meshgrid(x_args, y_args)
    x_grid = x_grid.reshape(-1, 1)
    y_grid = y_grid.reshape(-1, 1)
    return jp.concatenate([x_grid, y_grid], axis=1)


def assemble(H, W, H_tiles, W_tiles, image_blocks):
    tile_H = H // H_tiles
    tile_W = W // W_tiles
    shape = (H_tiles, W_tiles, tile_H, tile_W, -1)
    tiles = jp.reshape(image_blocks, shape)
    rows = [jp.hstack(tiles[row_arg]) for row_arg in range(H_tiles)]
    return jp.vstack(rows)
