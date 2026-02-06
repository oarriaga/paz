from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jp

from paz.graphics.constants import FARAWAY
from paz.graphics.geometry import (
    compute_reflections_dot_eye,
    compute_hits_to_light,
)
import paz


EPSILON = 1e-8

Mesh = namedtuple(
    "Mesh",
    ["vertices", "vertex_colors", "transform", "material", "faces", "edges"],
)


def render(image_shape, world_to_camera, rays, meshes, mask, lights):
    _render_mesh = partial(render_mesh, lights, *rays)
    hit_masks, depths, colors = jax.vmap(_render_mesh)(meshes)
    hit_masks, depths = mask_out_mesh(mask, hit_masks, depths)
    num_meshes, num_triangles, num_pixels = depths.shape
    depths = jp.reshape(depths, (num_meshes * num_triangles, num_pixels, 1))
    colors = jp.reshape(colors, (num_meshes * num_triangles, num_pixels, 3))
    return postprocess(hit_masks, depths, colors, world_to_camera, rays, image_shape)


def render_mesh(lights, ray_origins, ray_directions, mesh):
    hit_mask, depths = intersect_mesh(mesh, ray_origins, ray_directions)
    depths = jp.expand_dims(depths, axis=-1)
    points = compute_position(ray_origins, ray_directions, depths)
    normals = compute_normals(mesh.vertices, mesh.faces, mesh.transform, points)
    eyes = -ray_directions
    colors = compute_mesh_colors(mesh, lights, points, normals, eyes)
    hit_mask = compute_scene_hit_mask(hit_mask)
    return hit_mask, jp.squeeze(depths, axis=-1), colors


def postprocess(hit_masks, depths, colors, world_to_camera, rays, image_shape):
    H, W = image_shape
    scene_hit_mask = jp.any(hit_masks, axis=0)
    scene_colors = select_triangle_color(depths[..., 0], colors)
    image = to_color_image(scene_hit_mask, scene_colors, H, W)
    min_depths = jp.min(depths[..., 0], axis=0)
    depth = to_depth_image(scene_hit_mask, min_depths, world_to_camera, rays, H, W)
    return image, depth


def mask_out_mesh(mask, hit_masks, depths):
    mask = jp.expand_dims(mask, 1)
    hit_masks = jp.where(mask, hit_masks, False)
    depths = jp.where(jp.expand_dims(mask, 1), depths, 1e6)
    return hit_masks, depths


def intersect_mesh(mesh, ray_origins, ray_directions):
    world_to_shape = jp.linalg.inv(mesh.transform)
    rays = paz.algebra.transform_rays(world_to_shape, ray_origins, ray_directions)
    return intersect_canonical_mesh(mesh.vertices, mesh.faces, *rays)


def intersect_canonical_mesh(vertices, faces, ray_origins, ray_directions):
    edges_AB, edges_AC, points_A = build_edges(vertices, faces)
    f, directions_cross_edges_AC = compute_f(edges_AC, edges_AB, ray_directions)
    points_1_to_origin = ray_origins - points_A
    u = f * paz.algebra.dot(points_1_to_origin, directions_cross_edges_AC)
    origins_cross_edge_1 = jp.cross(points_1_to_origin, edges_AB)
    v = f * paz.algebra.dot(ray_directions, origins_cross_edge_1)
    hit_mask_u = jp.logical_not(jp.logical_or(u < 0.0, u > 1.0))
    hit_mask_v = jp.logical_not(jp.logical_or(v < 0.0, (u + v) > 1.0))
    hit_mask = jp.logical_and(hit_mask_u, hit_mask_v)
    depth = f * paz.algebra.dot(edges_AC, origins_cross_edge_1)
    depth = jp.where(hit_mask, depth, FARAWAY)
    return hit_mask, depth


def compute_normals(vertices, faces, transform, world_points):
    inverse = jp.linalg.inv(transform)
    shape_points = transform_points(inverse, world_points)
    normals = compute_canonical_normals(vertices, faces, shape_points)
    normals = transform_points(inverse.T, normals)
    normals = paz.algebra.normalize(normals)
    return normals


def compute_mesh_colors(mesh, lights, points, normals, eyes):
    face_colors = vertex_colors_to_face_colors(mesh.faces, mesh.vertex_colors)
    colors = []
    material = mesh.material
    for light in lights:
        ambient = compute_ambient(material.ambient, face_colors, light, points)
        diffuse = compute_diffuse(
            material.diffuse, face_colors, light, points, normals
        )
        specular = compute_specular(
            material.specular, material.shininess, eyes, light, points, normals
        )
        color = ambient + diffuse + specular
        colors.append(color)
    colors = jp.sum(jp.array(colors), axis=0)
    return colors


def compute_ambient(ambient, color, light, points):
    base_color = compute_base_color(color, light.intensity, points)
    return base_color * ambient


def compute_diffuse(diffuse, color, light, points, normals):
    hits_to_light = compute_hits_to_light(light.position, points)
    lambertian = paz.algebra.dot(hits_to_light, normals)
    lambertian = jp.maximum(lambertian, 0.0)
    lambertian = jp.expand_dims(lambertian, -1)
    base_color = compute_base_color(color, light.intensity, points)
    return base_color * diffuse * lambertian


def compute_specular(specular, shininess, eyes, light, points, normals):
    reflections = compute_reflections_dot_eye(light, points, normals, eyes)
    factor = jp.expand_dims(jp.power(reflections, shininess), -1)
    specular_color = light.intensity * specular * factor
    return specular_color


def compute_base_color(color, intensity, points):
    return color * intensity


def vertex_colors_to_face_colors(faces, vertex_colors):
    face_colors = jp.mean(vertex_colors[faces], axis=1)
    face_colors = jp.expand_dims(face_colors, axis=1)
    return face_colors


def compute_position(ray_origins, ray_directions, depths):
    ray_directions = jp.expand_dims(ray_directions, axis=0)
    ray_origins = jp.expand_dims(ray_origins, axis=0)
    positions = ray_origins + (depths * ray_directions)
    return positions


def compute_canonical_normals(vertices, faces, shape_points):
    points_A, points_B, points_C = extract_points(vertices, faces)
    edges_AB = points_B - points_A
    edges_AC = points_C - points_A
    normals = jp.cross(edges_AC, edges_AB)
    normals = paz.algebra.normalize(normals)
    num_rays = shape_points.shape[1]
    normals = jp.expand_dims(normals, 1)
    normals = jp.repeat(normals, num_rays, axis=1)
    return normals


def transform_points(affine_transform, points):
    ones = jp.ones((*points.shape[:2], 1))
    points = jp.concatenate([points, ones], axis=-1)
    points = jp.swapaxes(points, 1, 2)
    points = jp.matmul(affine_transform, points)
    points = jp.swapaxes(points, 2, 1)
    return points[:, :, :3]


def build_edges(vertices, faces):
    points_A, points_B, points_C = extract_points(vertices, faces)
    edges_AB = points_B - points_A
    edges_AC = points_C - points_A
    edges_AC = jp.expand_dims(edges_AC, axis=1)
    edges_AB = jp.expand_dims(edges_AB, axis=1)
    points_A = jp.expand_dims(points_A, axis=1)
    return edges_AC, edges_AB, points_A


def extract_points(vertices, faces):
    points_A = vertices[faces[:, 0]]
    points_B = vertices[faces[:, 1]]
    points_C = vertices[faces[:, 2]]
    return points_A, points_B, points_C


def compute_f(edges_AC, edges_AB, ray_directions):
    directions_cross_edges_AC = jp.cross(ray_directions, edges_AC)
    determinants = paz.algebra.dot(edges_AB, directions_cross_edges_AC)
    f = 1.0 / (determinants + EPSILON)
    return f, directions_cross_edges_AC


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


def tile_render(tile_shape, y_FOV, H, W, world_to_camera, meshes, mask, lights):
    num_H_tiles, num_W_tiles = tile_shape
    assert_exact_tile_side(H, num_H_tiles)
    assert_exact_tile_side(W, num_W_tiles)
    args = (H, W, num_H_tiles, num_W_tiles, y_FOV, world_to_camera, meshes, mask, lights)
    _render = partial(render_tile, *args)
    tile_coordinates = make_tile_coordinates(num_H_tiles, num_W_tiles)
    image, depth = jax.lax.scan(_render, None, tile_coordinates)[1]
    image = assemble(H, W, num_H_tiles, num_W_tiles, image)
    depth = assemble(H, W, num_H_tiles, num_W_tiles, depth)[..., 0]
    return image, depth


def render_tile(
    H, W, num_H_tiles, num_W_tiles, y_FOV,
    world_to_camera, meshes, mask, lights, carry, tile_arg,
):
    camera_to_world = jp.linalg.inv(world_to_camera)
    rays = build_tile_rays(
        H, W, num_H_tiles, num_W_tiles, y_FOV, camera_to_world, tile_arg,
    )
    tile_shape = (H // num_H_tiles, W // num_W_tiles)
    tile = render(tile_shape, world_to_camera, rays, meshes, mask, lights)
    return carry, tile


def assemble(H, W, num_H_tiles, num_W_tiles, image_blocks):
    tile_H = H // num_H_tiles
    tile_W = W // num_W_tiles
    shape = (num_H_tiles, num_W_tiles, tile_H, tile_W, -1)
    tiles = jp.reshape(image_blocks, shape)
    rows = [jp.hstack(tiles[row_arg]) for row_arg in range(num_H_tiles)]
    return jp.vstack(rows)


def build_tile_rays(H, W, num_H_tiles, num_W_tiles, y_FOV, camera_to_world, tile_arg):
    aspect_ratio = paz.graphics.camera.compute_aspect_ratio(H, W)
    H_world, W_world = paz.graphics.camera.compute_image_sizes(y_FOV, aspect_ratio)
    half_W = W_world / 2
    half_H = H_world / 2
    pixel_size = paz.graphics.camera.compute_pixel_size(W_world, W)
    tile_H = H // num_H_tiles
    tile_W = W // num_W_tiles
    ray_targets = make_ray_targets(
        tile_H, tile_W, pixel_size, half_W, half_H, tile_arg,
    )
    ray_origins = make_ray_origins(tile_H, tile_W)
    return transform_tile_rays(camera_to_world, ray_origins, ray_targets)


def make_ray_targets(tile_H, tile_W, pixel_size, half_W, half_H, tile_arg):
    W_tile_arg, H_tile_arg = tile_arg
    W_start = tile_W * W_tile_arg
    W_final = tile_W * (W_tile_arg + 1)
    H_start = tile_H * H_tile_arg
    H_final = tile_H * (H_tile_arg + 1)
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


def make_tile_coordinates(num_H_tiles, num_W_tiles):
    x_args = jp.arange(num_W_tiles)
    y_args = jp.arange(num_H_tiles)
    x_grid, y_grid = jp.meshgrid(x_args, y_args)
    x_grid = x_grid.reshape(-1, 1)
    y_grid = y_grid.reshape(-1, 1)
    return jp.concatenate([x_grid, y_grid], axis=1)


def transform_tile_rays(camera_to_world, ray_origins, ray_targets):
    origins = jp.matmul(camera_to_world, ray_origins.T).T
    targets = jp.matmul(camera_to_world, ray_targets.T).T
    directions = targets[:, :3] - origins[:, :3]
    directions = paz.algebra.normalize(directions)
    return origins[:, :3], directions


def assert_exact_tile_side(image_size, tile_size):
    if (image_size / tile_size) % 1 != 0:
        raise ValueError("tile size must divide image size without a residual")


def fill_bottom_with_last(x, total_size):
    if len(x) > total_size:
        raise ValueError("`x` length should be smaller than `total_size`")
    missing_size = total_size - len(x)
    last = x[-2:-1]
    repeated_last = jp.repeat(last, missing_size, axis=0)
    return jp.concatenate([x, repeated_last], axis=0)


def fill_mesh(mesh_to_fill, num_vertices, num_faces, num_edges):
    vertices = fill_bottom_with_last(mesh_to_fill.vertices, num_vertices)
    faces = fill_bottom_with_last(mesh_to_fill.faces, num_faces)
    edges = fill_bottom_with_last(mesh_to_fill.edges, num_edges)
    vertex_colors = fill_bottom_with_last(
        mesh_to_fill.vertex_colors, num_vertices
    )
    transform, material = mesh_to_fill.transform, mesh_to_fill.material
    return Mesh(vertices, vertex_colors, transform, material, faces, edges)


def merge_meshes(*meshes):
    max_vertices = max(m.vertices.shape[0] for m in meshes)
    max_faces = max(m.faces.shape[0] for m in meshes)
    max_edges = max(m.edges.shape[0] for m in meshes)
    filled = [fill_mesh(m, max_vertices, max_faces, max_edges) for m in meshes]
    batched = jax.tree.map(lambda *args: jp.stack(args), *filled)
    mask = jp.ones(len(meshes), dtype=bool)
    return batched, mask


def build_cube(size=1.0):
    import trimesh
    import numpy as onp
    mesh = trimesh.creation.box(extents=[size, size, size])
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    faces = jp.concatenate([faces[:, 0:1], faces[:, 2:3], faces[:, 1:2]], axis=1)
    edges = jp.array(mesh.edges.view(onp.ndarray))
    return vertices, faces, edges


def build_sphere(radius=1.0, subdivisions=3):
    import trimesh
    import numpy as onp
    mesh = trimesh.creation.icosphere(subdivisions, radius)
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    faces = jp.concatenate([faces[:, 0:1], faces[:, 2:3], faces[:, 1:2]], axis=1)
    edges = jp.array(mesh.edges.view(onp.ndarray))
    return vertices, faces, edges


def load_mesh(filepath):
    import trimesh
    import numpy as onp
    mesh = trimesh.load(filepath)
    vertices = jp.array(mesh.vertices.view(onp.ndarray))
    faces = jp.array(mesh.faces.view(onp.ndarray))
    vertex_colors = mesh.visual.vertex_colors[:, :3]
    vertex_colors = jp.array(vertex_colors.view(onp.ndarray))
    vertex_colors = vertex_colors / 255.0
    return vertices, faces, vertex_colors
