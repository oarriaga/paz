import jax
import jax.numpy as jp

import paz


def compute_triangle_albedo(triangles, face_index, u, v):
    args = triangles.vertex_colors, triangles.faces, face_index, u, v
    vertex_colors = paz.graphics.mesh.interpolate_for_hits(*args)
    if has_texture(triangles):
        texture = compute_texture_albedo(triangles, face_index, u, v)
        use_uv = jp.any(triangles.vertex_uvs != 0.0)
        albedo = jp.where(use_uv, texture, vertex_colors)
    else:
        albedo = vertex_colors
    return albedo


def has_texture(triangles):
    image = triangles.patterns.image
    return (image.shape[1] > 1) or (image.shape[2] > 1)


def compute_texture_albedo(triangles, face_index, u, v):
    args = triangles.vertex_uvs, triangles.faces, face_index, u, v
    texture_uv = paz.graphics.mesh.interpolate_for_hits(*args)
    sample = paz.lock(sample_texture, texture_uv)
    colors = jax.vmap(sample)(triangles.patterns.image)
    primitive = triangles.primitive_index[face_index]
    colors = take_per_primitive(colors, primitive)
    return colors + triangles.materials.color[primitive]


def sample_texture(image, texture_uv):
    sample = paz.graphics.patterns.image.compute_image_colors_bilinear
    return sample(texture_uv[:, 0:1], texture_uv[:, 1:2], image)


def take_per_primitive(colors, primitive):
    indices = primitive[None, :, None]
    return jp.squeeze(jp.take_along_axis(colors, indices, axis=0), axis=0)


def compute_shape_albedo(shape, material, points):
    pattern_colors = compute_pattern_colors(shape, points)
    material_colors = jp.full_like(points, material.color)
    return pattern_colors + material_colors


def compute_pattern_colors(shape, points):
    args = shape.pattern.transform, shape.transform, points
    pattern_points = compute_points_in_pattern(*args)
    cases = [
        paz.graphics.patterns.empty.compute_colors,
        paz.graphics.patterns.spherical.compute_colors,
        paz.graphics.patterns.planar.compute_colors,
        paz.graphics.patterns.cylindrical.compute_colors,
    ]
    switch_args = pattern_points, shape.pattern.image
    return jax.lax.switch(shape.pattern.type, cases, *switch_args)


def compute_points_in_pattern(pattern_transform, shape_transform, points):
    world_to_shape = jp.linalg.inv(shape_transform)
    points_shape = paz.algebra.transform_points(world_to_shape, points)
    shape_to_pattern = jp.linalg.inv(pattern_transform)
    return paz.algebra.transform_points(shape_to_pattern, points_shape)
