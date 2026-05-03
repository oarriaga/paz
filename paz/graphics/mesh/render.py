from collections import namedtuple

import jax
import jax.numpy as jp

import paz

from .types import normalize_mesh_batch
from .intersect import intersect_mesh
from .geometry import compute_normals_for_hits
from .shading import compute_colors_for_hits
from .tile import assemble, assert_exact_tile_side
from .tile import build_tile_rays, make_tile_coordinates

RENDER_ARGS = "shape y_FOV pose meshes mask lights tiles chunk_size"
RenderArgs = namedtuple("RenderArgs", RENDER_ARGS)


def render(shape, y_FOV, pose, meshes, mask, lights, tiles, chunk_size):
    meshes = normalize_mesh_batch(meshes)
    args = shape, y_FOV, pose, meshes, mask, lights, tiles, chunk_size
    args = RenderArgs(*args)
    image, depth = _scan_tiles(args, _render_tile)
    return _assemble_image(args, image), _assemble_depth(args, depth)


def render_masks(shape, y_FOV, pose, meshes, lights, depth, tiles, chunk_size):
    meshes = normalize_mesh_batch(meshes)
    min_depth, max_depth = depth
    num_meshes = len(meshes.vertices)
    masks = []
    for arg in range(num_meshes):
        mask = jp.zeros(num_meshes, dtype=bool).at[arg].set(True)
        args = shape, y_FOV, pose, meshes, mask, lights, tiles, chunk_size
        args = RenderArgs(*args)
        depth_image = _scan_tiles(args, _render_depth_tile)
        depth_image = _assemble_depth(args, depth_image)
        soft = paz.depth.to_soft_mask(depth_image, min_depth, max_depth)
        masks.append(jp.expand_dims(soft, axis=-1))
    return jp.stack(masks)


def _scan_tiles(args, render_step):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    assert_exact_tile_side(H, H_tiles)
    assert_exact_tile_side(W, W_tiles)
    coordinates = make_tile_coordinates(H_tiles, W_tiles)
    render_step = paz.lock(render_step, args)
    return jax.lax.scan(render_step, None, coordinates)[1]


def _render_tile(carry, tile_arg, args):
    rays, shape = _build_tile_rays(args, tile_arg)
    render_args = shape, args.pose, rays, args.meshes
    render_args = render_args + (args.mask, args.lights, args.chunk_size)
    image, depth = _render_rays(*render_args)
    return carry, (image, jp.expand_dims(depth, -1))


def _render_depth_tile(carry, tile_arg, args):
    rays, shape = _build_tile_rays(args, tile_arg)
    render_args = shape, args.pose, rays, args.meshes
    render_args = render_args + (args.mask, args.chunk_size)
    depth = _render_depth_rays(*render_args)
    return carry, jp.expand_dims(depth, -1)


def _build_tile_rays(args, tile_arg):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    camera_to_world = jp.linalg.inv(args.pose)
    ray_args = H, W, H_tiles, W_tiles, args.y_FOV, camera_to_world
    rays = build_tile_rays(*ray_args, tile_arg)
    return rays, (H // H_tiles, W // W_tiles)


def _assemble_image(args, image):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    return assemble(H, W, H_tiles, W_tiles, image)


def _assemble_depth(args, depth):
    H, W = args.shape
    H_tiles, W_tiles = args.tiles
    return assemble(H, W, H_tiles, W_tiles, depth)[..., 0]


def _render_rays(shape, pose, rays, meshes, mask, lights, chunk_size):
    render_fn = _checkpointed_render(lights, rays, chunk_size)
    hit_masks, depths, colors = jax.vmap(render_fn)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    depths = jp.expand_dims(depths, -1)
    args = hit_masks, depths, colors, pose, rays, shape
    return postprocess(*args)


def _render_depth_rays(shape, pose, rays, meshes, mask, chunk_size):
    render_fn = _checkpointed_depth(rays, chunk_size)
    hit_masks, depths = jax.vmap(render_fn)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    depths = jp.expand_dims(depths, -1)
    args = hit_masks, depths, pose, rays, shape
    return _postprocess_depth(*args)


def _checkpointed_render(lights, rays, chunk_size):
    @jax.checkpoint
    def fn(mesh):
        return _render_mesh(lights, *rays, mesh, chunk_size)
    return fn


def _checkpointed_depth(rays, chunk_size):
    @jax.checkpoint
    def fn(mesh):
        return _render_mesh_depth(*rays, mesh, chunk_size)
    return fn


def _render_mesh(lights, ray_origins, ray_directions, mesh, chunk_size=1024):
    rays = (ray_origins, ray_directions)
    result = intersect_mesh(mesh, *rays, chunk_size=chunk_size)
    hit_mask, depth, u, v, face_idx = result
    depth_3d = jp.expand_dims(depth, -1)
    points = ray_origins + depth_3d * ray_directions
    args = (mesh.vertices, mesh.faces, mesh.transform, face_idx)
    normals = compute_normals_for_hits(*args)
    eyes = -ray_directions
    args = (mesh, lights, points, normals, eyes, face_idx, u, v)
    colors = compute_colors_for_hits(*args)
    return hit_mask, depth, colors


def _render_mesh_depth(ray_origins, ray_directions, mesh, chunk_size=1024):
    rays = (ray_origins, ray_directions)
    result = intersect_mesh(mesh, *rays, chunk_size=chunk_size)
    hit_mask, depth, _, _, _ = result
    return hit_mask, depth


def postprocess(hit_masks, depths, colors, world_to_camera, rays, image_shape):
    H, W = image_shape
    scene_hit_mask = jp.any(hit_masks, axis=0)
    scene_colors = select_closest_color(depths[..., 0], colors)
    image = to_color_image(scene_hit_mask, scene_colors, H, W)
    min_depths = jp.min(depths[..., 0], axis=0)
    args = scene_hit_mask, min_depths, world_to_camera, rays, H, W
    depth = to_depth_image(*args)
    return image, depth


def _postprocess_depth(hit_masks, depths, world_to_camera, rays, image_shape):
    H, W = image_shape
    scene_hit_mask = jp.any(hit_masks, axis=0)
    min_depths = jp.min(depths[..., 0], axis=0)
    args = scene_hit_mask, min_depths, world_to_camera, rays, H, W
    return to_depth_image(*args)


def mask_out_mesh(mask, hit_masks, depths):
    mask = jp.expand_dims(mask, 1)
    hit_masks = jp.where(mask, hit_masks, False)
    depths = jp.where(mask, depths, 1e6)
    return hit_masks, depths


def select_closest_color(depths, colors):
    arg_depths = jp.argmin(depths, axis=0)
    idx = jp.expand_dims(arg_depths, 0)
    colors = jp.take_along_axis(colors, jp.expand_dims(idx, -1), 0)
    return jp.squeeze(colors, axis=0)


def to_color_image(hit_mask, colors, H, W):
    image = jp.where(hit_mask[:, None], colors, 1.0)
    image = jp.clip(image, 0, 1)
    image = jp.reshape(image, (H, W, 3))
    return image


def to_depth_image(hit_mask, depths, world_to_camera, rays, H, W):
    ray_origins, ray_directions = rays
    points = ray_origins + jp.expand_dims(depths, -1) * ray_directions
    points = paz.algebra.transform_points(world_to_camera, points)
    world_depths = -points[:, -1]
    masked_depths = jp.where(hit_mask, world_depths, 0.0)
    return jp.reshape(masked_depths, (H, W))
