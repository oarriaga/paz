import jax
import jax.numpy as jp

import paz

from paz.graphics.composite import postprocess


def scan(shape, tiles, tile_step):
    H, W = shape
    H_tiles, W_tiles = tiles
    paz.graphics.mesh.assert_exact_tile_side(H, H_tiles)
    paz.graphics.mesh.assert_exact_tile_side(W, W_tiles)
    coordinates = paz.graphics.mesh.make_tile_coordinates(H_tiles, W_tiles)
    return jax.lax.scan(tile_step, None, coordinates)[1]


def render_step(carry, tile_arg, shape, y_FOV, pose, tiles, render_rays):
    H, W = shape
    H_tiles, W_tiles = tiles
    camera_to_world = jp.linalg.inv(pose)
    tile_args = H, W, H_tiles, W_tiles, y_FOV, camera_to_world
    rays = paz.graphics.mesh.build_tile_rays(*tile_args, tile_arg)
    hit_mask, depth, color = render_rays(rays)
    tile_H, tile_W = H // H_tiles, W // W_tiles
    post_args = hit_mask, depth, color, pose, rays, tile_H, tile_W
    image, depth = postprocess(*post_args)
    return carry, (image, jp.expand_dims(depth, -1))


def assemble(shape, tiles, images):
    H, W = shape
    H_tiles, W_tiles = tiles
    return paz.graphics.mesh.assemble(H, W, H_tiles, W_tiles, images)
