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
    is_positive = jp.logical_or(depths_A > EPSILON, depths_B > EPSILON)
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
    choose_A = jp.logical_and((depths_A > EPSILON), (depths_A < depths_B))
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
    dots = jp.expand_dims(paz.algebra.dot(light_directions, normals), -1)
    return light_directions - (normals * 2 * dots)


def refract(incident, normals, n1, n2):
    # n1: current refractive index (scalar or array)
    # n2: target refractive index (scalar or array)
    # incident: (N, 3)
    # normals: (N, 3) - assumed to be pointing against incident (cos_theta > 0)

    eta = n1 / n2
    # Expand dims for broadcasting if n1/n2 are scalars vs arrays
    # Actually usually they will be arrays matching incident length in the renderer loop
    eta = jp.expand_dims(eta, -1)

    dot_ni = paz.algebra.dot(incident, normals)
    cos_i = -dot_ni
    cos_i = jp.expand_dims(cos_i, -1)

    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    discriminant = 1.0 - sin2_t
    is_tir = discriminant < 0.0

    sqrt_discriminant = jp.sqrt(jp.where(is_tir, 0.0, discriminant))
    refracted = (eta * incident) + ((eta * cos_i - sqrt_discriminant) * normals)

    reflected = reflect(incident, normals)
    return jp.where(is_tir, reflected, refracted)


def compute_reflections_dot_eye(light, points, normals, eye):
    hits_to_light = compute_hits_to_light(light.position, points)
    reflections = reflect(-hits_to_light, normals)
    eye = paz.algebra.normalize(eye)
    reflections_dot_eye = paz.algebra.dot(reflections, eye)
    reflections_dot_eye = jp.clip(reflections_dot_eye, 0.0, 1.0)
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
