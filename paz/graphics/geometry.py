import jax.numpy as jp
import paz
from paz.graphics.constants import FARAWAY, EPSILON


def compute_quadratic_is_hit(depths_A, depths_B, is_valid):
    """Computes is hit mask from quadratic depths solution

    # Arguments
        depths_A: Array
        depths_B: Array
        is_Valid: Array

    # Returns
        is_hit: Array

    # Notes
        TODO: check if it should not be >= instead of just >
    """
    is_positive = jp.logical_or(depths_A > 0, depths_B > 0)
    is_hit = jp.logical_and(is_valid, is_positive)
    return is_hit


def apply_hit_mask(hit_mask, depths):
    """Applies hit mask to depths

    # Arguments
        hit_mask: Array
        depths: Array

    # Returns
        depths: Array
    """
    depth = jp.where(hit_mask, depths, FARAWAY)
    return depth


def compute_quadratic_depths(depths_A, depths_B):
    """Computes closest positive depth from quadratic solutions

    # Arguments
        depth_A: Array
        depth_B: Array
        hit_mask: Array

    # Returns
        depth: Array
    """
    choose_A = jp.logical_and((depths_A > 0), (depths_A < depths_B))
    depth = jp.where(choose_A, depths_A, depths_B)
    return depth


def replace_misses(depth, hit_mask):
    """Replaces depths where rays miss hit with faraway values

    # Arguments
        depth: Array
        hit_mask:

    # Return
        depth: Array
    """
    depth = jp.where(hit_mask, depth, FARAWAY)
    return depth


def compute_points3D(origins, directions, depth):
    """Compute positions of 3D points i.e. pointcloud

    # Arguments
        ray_origins: Array (num_rays, 3)
        ray_directions: Array (num_rays, 3)
        depth: Array (???)

    # Returns
        Array (num_rays, 3)
    """
    position = origins + (depth * directions)
    return position


def compute_hits_to_light(light_position, hits):
    hits_to_light = light_position - hits
    hits_to_light = paz.algebra.normalize(hits_to_light)
    return hits_to_light


def reflect(light_directions, normals):
    return light_directions - (
        normals
        * 2
        * jp.expand_dims(paz.algebra.dot(light_directions, normals), -1)
    )


def compute_reflections_dot_eye(light, points, normals, eye):
    hits_to_light = compute_hits_to_light(light.position, points)
    reflections = reflect(-hits_to_light, normals)
    reflections_dot_eye = paz.algebra.dot(reflections, eye)
    reflections_dot_eye = jp.maximum(reflections_dot_eye, 0.0)
    return reflections_dot_eye


def transform_rays(affine_transform, ray_origins, ray_directions):
    ray_origin = paz.algebra.transform_points(affine_transform, ray_origins)
    ray_directions = paz.algebra.transform_vectors(
        affine_transform, ray_directions
    )
    return ray_origin, ray_directions


def sort_depths(depths):
    # TODO check if sorting is decreasing 10x performance.
    depths = jp.vstack(depths)
    depths = jp.sort(depths, axis=0)
    return depths
