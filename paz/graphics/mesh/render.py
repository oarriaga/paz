from functools import partial

import jax
import jax.numpy as jp

import paz

from .types import normalize_mesh_batch
from .intersect import intersect_mesh
from .geometry import compute_position, compute_normals
from .shading import compute_mesh_colors


def render(image_shape, world_to_camera, rays, meshes, mask, lights):
    meshes = normalize_mesh_batch(meshes)
    _render_mesh = partial(render_mesh, lights, *rays)
    hit_masks, depths, colors = jax.vmap(_render_mesh)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    num_meshes, num_triangles, num_pixels = depths.shape
    depths = jp.reshape(depths, (num_meshes * num_triangles, num_pixels, 1))
    colors = jp.reshape(colors, (num_meshes * num_triangles, num_pixels, 3))
    args = (hit_masks, depths, colors, world_to_camera, rays, image_shape)
    return postprocess(*args)


def render_depth(image_shape, world_to_camera, rays, meshes, mask, lights):
    meshes = normalize_mesh_batch(meshes)
    _render_mesh_depth = partial(render_mesh_depth, lights, *rays)
    hit_masks, depths = jax.vmap(_render_mesh_depth)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    num_meshes, num_triangles, num_pixels = depths.shape
    depths = jp.reshape(depths, (num_meshes * num_triangles, num_pixels, 1))
    args = (hit_masks, depths, world_to_camera, rays, image_shape)
    return postprocess_depth(*args)


def render_mesh(lights, ray_origins, ray_directions, mesh):
    rays = (ray_origins, ray_directions)
    hit_mask, depths, barycentric_u, barycentric_v = intersect_mesh(mesh, *rays)
    depths = jp.expand_dims(depths, axis=-1)
    points = compute_position(ray_origins, ray_directions, depths)
    normals = compute_normals(mesh.vertices, mesh.faces, mesh.transform, points)
    eyes = -ray_directions
    uv = (barycentric_u, barycentric_v)
    colors = compute_mesh_colors(mesh, lights, points, normals, eyes, *uv)
    hit_mask = compute_scene_hit_mask(hit_mask)
    return hit_mask, jp.squeeze(depths, axis=-1), colors


def render_mesh_depth(_, ray_origins, ray_directions, mesh):
    rays = (ray_origins, ray_directions)
    hit_mask, depths, _, _ = intersect_mesh(mesh, *rays)
    hit_mask = compute_scene_hit_mask(hit_mask)
    return hit_mask, depths


def postprocess(hit_masks, depths, colors, world_to_camera, rays, image_shape):
    H, W = image_shape
    scene_hit_mask = jp.any(hit_masks, axis=0)
    scene_colors = select_triangle_color(depths[..., 0], colors)
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
    depths = jp.where(jp.expand_dims(mask, 1), depths, 1e6)
    return hit_masks, depths


def select_triangle_color(depths, colors):
    arg_depths = jp.argmin(depths, axis=0)
    arg_depths = jp.expand_dims(arg_depths, 0)
    colors = jp.take_along_axis(colors, jp.expand_dims(arg_depths, -1), axis=0)
    colors = jp.squeeze(colors, axis=0)
    return colors


def compute_scene_hit_mask(hit_masks):
    hit_masks = jp.array(hit_masks)
    hit_mask = jp.sum(hit_masks, axis=0)
    hit_mask = hit_mask.astype(bool)
    return hit_mask


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
