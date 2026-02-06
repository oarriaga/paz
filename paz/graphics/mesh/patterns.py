import jax
import jax.numpy as jp

from paz.graphics.constants import NO_PATTERN
import paz

from .geometry import transform_points


def compute_mesh_vertex_uvs(vertices, pattern):
    if pattern is None:
        return None
    pattern_inverse = jp.linalg.inv(pattern.transform)
    pattern_points = paz.algebra.transform_points(pattern_inverse, vertices)

    def no_pattern(points):
        return jp.zeros((points.shape[0], 2))

    def spherical(points):
        u, v = paz.graphics.patterns.spherical.spherical_map(points)
        return jp.concatenate([u, v], axis=1)

    def planar(points):
        u, v = paz.graphics.patterns.planar.planar_map(points)
        return jp.concatenate([u, v], axis=1)

    def cylindrical(points):
        u, v = paz.graphics.patterns.cylindrical.cylindrical_map(points)
        return jp.concatenate([u, v], axis=1)

    cases = [no_pattern, spherical, planar, cylindrical]
    return jax.lax.switch(pattern.type, cases, pattern_points)


def interpolate_triangle_values(values, faces, barycentric_u, barycentric_v):
    value_A = values[faces[:, 0]]
    value_B = values[faces[:, 1]]
    value_C = values[faces[:, 2]]
    weight_A = 1.0 - barycentric_u - barycentric_v
    weight_A = jp.expand_dims(weight_A, -1)
    weight_B = jp.expand_dims(barycentric_u, -1)
    weight_C = jp.expand_dims(barycentric_v, -1)
    return (
        value_A[:, None, :] * weight_A
        + value_B[:, None, :] * weight_B
        + value_C[:, None, :] * weight_C
    )


def compute_mesh_pattern_colors_from_uv(mesh, barycentric_u, barycentric_v):
    args = (mesh.vertex_uvs, mesh.faces, barycentric_u, barycentric_v)
    uv = interpolate_triangle_values(*args)
    uv_flat = jp.reshape(uv, (-1, 2))
    u = uv_flat[:, 0:1]
    v = uv_flat[:, 1:2]
    colors_flat = paz.graphics.patterns.image.compute_image_colors_bilinear(
        u, v, mesh.pattern.image
    )
    colors = jp.reshape(colors_flat, (uv.shape[0], uv.shape[1], 3))
    return colors


def compute_mesh_pattern_colors_from_points(mesh, points):
    world_to_shape = jp.linalg.inv(mesh.transform)
    shape_points = transform_points(world_to_shape, points)
    pattern_inverse = jp.linalg.inv(mesh.pattern.transform)
    pattern_points = transform_points(pattern_inverse, shape_points)
    num_faces, num_rays = pattern_points.shape[:2]
    pattern_points_flat = jp.reshape(pattern_points, (-1, 3))

    def no_pattern(points_arg):
        return jp.zeros((points_arg.shape[0], 3))

    def spherical(points_arg):
        u, v = paz.graphics.patterns.spherical.spherical_map(points_arg)
        args = (u, v, mesh.pattern.image)
        return paz.graphics.patterns.image.compute_image_colors_bilinear(*args)

    def planar(points_arg):
        u, v = paz.graphics.patterns.planar.planar_map(points_arg)
        args = (u, v, mesh.pattern.image)
        return paz.graphics.patterns.image.compute_image_colors_bilinear(*args)

    def cylindrical(points_arg):
        u, v = paz.graphics.patterns.cylindrical.cylindrical_map(points_arg)
        args = (u, v, mesh.pattern.image)
        return paz.graphics.patterns.image.compute_image_colors_bilinear(*args)

    cases = [no_pattern, spherical, planar, cylindrical]
    colors_flat = jax.lax.switch(mesh.pattern.type, cases, pattern_points_flat)
    return jp.reshape(colors_flat, (num_faces, num_rays, 3))


def compute_mesh_base_colors(mesh, points, barycentric_u, barycentric_v):
    uv = (barycentric_u, barycentric_v)
    has_pattern = jp.any(mesh.pattern.type != NO_PATTERN)
    has_vertex_uvs = mesh.vertex_uvs is not None
    if has_vertex_uvs:
        has_vertex_uvs = jp.any(mesh.vertex_uvs != 0.0)
    has_image = (mesh.pattern.image.shape[0] > 1) or (mesh.pattern.image.shape[1] > 1)  # fmt: skip

    def with_pattern_no_uv(_):
        pattern_colors = compute_mesh_pattern_colors_from_points(mesh, points)
        return pattern_colors + mesh.material.color

    def no_pattern(_):
        use_uv = has_vertex_uvs & has_image

        def with_uv(_):
            pattern_colors = compute_mesh_pattern_colors_from_uv(mesh, *uv)
            return pattern_colors + mesh.material.color

        def without_uv(_):
            if mesh.vertex_colors is not None:
                args = (mesh.vertex_colors, mesh.faces, *uv)
                return interpolate_triangle_values(*args)
            num_faces, num_rays = mesh.faces.shape[0], barycentric_u.shape[1]
            material_color = mesh.material.color
            return jp.broadcast_to(material_color, (num_faces, num_rays, 3))

        return jax.lax.cond(use_uv, with_uv, without_uv, None)

    return jax.lax.cond(has_pattern, with_pattern_no_uv, no_pattern, None)
