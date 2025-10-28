import jax.numpy as jp
import paz
from paz.graphics.geometry import apply_hit_mask
from paz.graphics.constants import EPSILON, FARAWAY


def check_axis(axis_origin, axis_direction):
    depth_min = -(axis_origin + 1) / axis_direction
    depth_max = -(axis_origin - 1) / axis_direction

    switch = depth_min > depth_max
    swapped_depth_min = jp.where(switch, depth_max, depth_min)
    swapped_depth_max = jp.where(switch, depth_min, depth_max)

    depth_min = swapped_depth_min
    depth_max = swapped_depth_max
    return depth_min, depth_max


def intersect_canonical_cube(ray_origins, ray_directions):
    x_origins, y_origins, z_origins = jp.split(ray_origins, 3, axis=1)
    directions = jp.split(ray_directions, 3, axis=1)
    x_directions, y_directions, z_directions = directions

    x_depth_min, x_depth_max = check_axis(x_origins, x_directions)
    y_depth_min, y_depth_max = check_axis(y_origins, y_directions)
    z_depth_min, z_depth_max = check_axis(z_origins, z_directions)

    depth_min = jp.maximum(x_depth_min, jp.maximum(y_depth_min, z_depth_min))
    depth_max = jp.minimum(x_depth_max, jp.minimum(y_depth_max, z_depth_max))

    hit_mask = jp.logical_and((depth_min > EPSILON), (depth_min < FARAWAY))
    max_bigger_than_min = depth_max > depth_min
    hit_mask = jp.logical_and(hit_mask, max_bigger_than_min)

    depth = depth_min
    hit_mask = jp.squeeze(hit_mask, 1)

    depth = jp.squeeze(depth, axis=1)
    depth = apply_hit_mask(hit_mask, depth)
    depth = jp.expand_dims(depth, axis=-1)

    depths = jp.vstack([depth_min[:, 0], depth_max[:, 0]])
    depths = paz.graphics.shapes.pad_depths(depths, hit_mask, 2)
    # TODO do apply_hit_mask for sorted depths as well
    return hit_mask, depths, depth


def compute_canonical_normals_cube(points):
    x_points, y_points, z_points = jp.split(points, 3, axis=1)
    abs_x_points = jp.abs(x_points)
    abs_y_points = jp.abs(y_points)
    abs_z_points = jp.abs(z_points)

    max_coordinates = jp.maximum(
        abs_x_points, jp.maximum(abs_y_points, abs_z_points)
    )

    zeros = jp.zeros((len(points), 1))
    x_normals = jp.hstack([x_points, zeros, zeros])
    y_normals = jp.hstack([zeros, y_points, zeros])
    z_normals = jp.hstack([zeros, zeros, z_points])

    normals = jp.zeros((len(points), 3))
    normals = jp.where(max_coordinates == abs_z_points, z_normals, normals)
    normals = jp.where(max_coordinates == abs_y_points, y_normals, normals)
    normals = jp.where(max_coordinates == abs_x_points, x_normals, normals)
    return normals
