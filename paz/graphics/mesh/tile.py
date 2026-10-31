import jax.numpy as jp

import paz


def assert_exact_tile_side(image_size, tile_size):
    if (image_size / tile_size) % 1 != 0:
        raise ValueError("tile size must divide image size without a residual")


def build_tile_rays(H, W, H_tiles, W_tiles, intrinsics, pose, tile_arg):
    tile_H = H // H_tiles
    tile_W = W // W_tiles
    pixels = build_tile_pixels(tile_H, tile_W, tile_arg)
    return paz.graphics.camera.build_pixel_rays(intrinsics, pixels, pose)


def build_tile_pixels(tile_H, tile_W, tile_arg):
    W_tile_arg, H_tile_arg = tile_arg
    start = jp.array([tile_W * W_tile_arg, tile_H * H_tile_arg])
    return paz.graphics.camera.build_pixel_grid(tile_H, tile_W) + start


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
