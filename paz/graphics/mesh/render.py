from functools import partial

import jax
import jax.numpy as jp

import paz

from .types import normalize_mesh_batch
from .intersect import intersect_mesh
from .geometry import compute_normals_for_hits
from .shading import compute_colors_for_hits


def render(image_shape, world_to_camera, rays, meshes, mask, lights):
    meshes = normalize_mesh_batch(meshes)
    _render_mesh = partial(render_mesh, lights, *rays)
    hit_masks, depths, colors = jax.vmap(_render_mesh)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    depths = jp.expand_dims(depths, -1)
    args = (hit_masks, depths, colors, world_to_camera, rays, image_shape)
    return postprocess(*args)


def render_depth(image_shape, world_to_camera, rays, meshes, mask, lights):
    meshes = normalize_mesh_batch(meshes)
    _render_depth = partial(render_mesh_depth, lights, *rays)
    hit_masks, depths = jax.vmap(_render_depth)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    depths = jp.expand_dims(depths, -1)
    args = (hit_masks, depths, world_to_camera, rays, image_shape)
    return postprocess_depth(*args)


@jax.checkpoint
def render_mesh(lights, ray_origins, ray_directions, mesh):
    rays = (ray_origins, ray_directions)
    hit_mask, depth, u, v, face_idx = intersect_mesh(mesh, *rays)
    depth_3d = jp.expand_dims(depth, -1)
    points = ray_origins + depth_3d * ray_directions
    args = (mesh.vertices, mesh.faces, mesh.transform, face_idx)
    normals = compute_normals_for_hits(*args)
    eyes = -ray_directions
    args = (mesh, lights, points, normals, eyes, face_idx, u, v)
    colors = compute_colors_for_hits(*args)
    return hit_mask, depth, colors


def render_mesh_depth(_, ray_origins, ray_directions, mesh):
    rays = (ray_origins, ray_directions)
    hit_mask, depth, _, _, _ = intersect_mesh(mesh, *rays)
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


def postprocess_depth(hit_masks, depths, world_to_camera, rays, image_shape):
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
