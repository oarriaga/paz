from functools import partial
import cv2
import numpy as np
import jax.numpy as jp
from jax import vmap

import paz
from paz.backend import algebra


def project_to_3D(camera_intrinsics, point2D, depth):
    u, v = point2D
    pixel = jp.array([u, v, 1.0])
    point3D = jp.linalg.inv(camera_intrinsics) @ pixel * -depth
    return point3D



def build_corner_rays(intrinsics, bounds=None, world_to_camera=None):
    """Build normalized rays from camera through corner pixels.

    Arguments:
        intrinsics: Camera intrinsics matrix (3x3).
        bounds: Optional tuple of (u_min, u_max, v_min, v_max) pixel coordinates.
            If None, uses full image bounds derived from intrinsics.
        world_to_camera: Optional transform matrix (4x4).

    Returns:
        Array of 4 normalized ray directions through the corner pixels.
    """
    if bounds is None:
        H, W = get_image_size(intrinsics)
        bounds = (0, W, 0, H)
    u_min, u_max, v_min, v_max = bounds
    corner_pixels = jp.array([
        [u_min, v_min],
        [u_max, v_min],
        [u_max, v_max],
        [u_min, v_max],
    ])
    world_to_camera = jp.eye(4) if world_to_camera is None else world_to_camera
    project = vmap(project_to_3D, in_axes=(None, 0, None))
    rays_camera = project(intrinsics, corner_pixels, 1.0)
    rotation_inverse = paz.SE3.get_rotation_matrix(world_to_camera).T
    rays_world = (rotation_inverse @ rays_camera.T).T
    norms = jp.linalg.norm(rays_world, axis=1, keepdims=True)
    return rays_world / norms


def intersect_frustum_plane(
    intrinsics, plane_normal, plane_centroid, bounds=None, world_to_camera=None
):
    """Compute frustum-plane intersection corners.

    Arguments:
        intrinsics: Camera intrinsics matrix (3x3).
        plane_normal: Normal vector of the plane.
        plane_centroid: A point on the plane.
        bounds: Optional tuple of (u_min, u_max, v_min, v_max) pixel coordinates.
            If None, uses full image bounds derived from intrinsics.
        world_to_camera: Optional transform matrix (4x4).

    Returns:
        Array of 4 intersection points where the frustum meets the plane.
    """
    rays = build_corner_rays(intrinsics, bounds, world_to_camera)
    ray_origins = jp.zeros_like(rays)
    return paz.plane.intersect_rays(
        plane_centroid, plane_normal, ray_origins, rays
    )


def shrink_frustum_corners(corners, margins):
    """Apply margins to shrink frustum corners inward."""
    left_margin, right_margin, near_margin, far_margin = jp.array(margins)
    far_left, far_right, near_right, near_left = corners
    left_depth_vector = far_left - near_left
    right_depth_vector = far_right - near_right
    near_left = near_left + left_depth_vector * near_margin
    near_right = near_right + right_depth_vector * near_margin
    far_left = far_left - left_depth_vector * far_margin
    far_right = far_right - right_depth_vector * far_margin
    near_edge = near_right - near_left
    far_edge = far_right - far_left
    return jp.stack(
        [
            far_left + far_edge * left_margin,
            far_right - far_edge * right_margin,
            near_right - near_edge * right_margin,
            near_left + near_edge * left_margin,
        ]
    )


def build_frustum_grid(corners, depth_steps, width_steps):
    """Build regular 2D grid of 3D points within frustum trapezoid."""
    far_left, far_right, near_right, near_left = corners
    depth_ratios = jp.linspace(0.0, 1.0, depth_steps)[:, None]
    width_ratios = jp.linspace(0.0, 1.0, width_steps)[None, :, None]
    left_edge = near_left + depth_ratios * (far_left - near_left)
    right_edge = near_right + depth_ratios * (far_right - near_right)
    grid = left_edge[:, jp.newaxis, :] + width_ratios * (
        right_edge[:, jp.newaxis, :] - left_edge[:, jp.newaxis, :]
    )
    return grid.reshape(-1, 3)


def get_image_size(camera_intrinsics):
    W = int(camera_intrinsics[0, 2] * 2.0)
    H = int(camera_intrinsics[1, 2] * 2.0)
    return H, W


def resize_intrinsics(camera_intrinsics, scale):
    return camera_intrinsics.copy().at[:2, :].mul(scale)


def project_to_2D(camera_matrix, point3D):
    homogenous_point3D = algebra.add_one(point3D)
    homogenous_point2D = jp.matmul(camera_matrix, homogenous_point3D)
    point2D = algebra.dehomogenize_coordinates(homogenous_point2D)
    return point2D


def make_camera_matrix(camera_intrinsics, camera_pose):
    # camera_intrinsics (3 x 3), camera_pose: (4 x 4)
    intrinsics = jp.concatenate([camera_intrinsics, jp.zeros((3, 1))], axis=1)
    camera_matrix = jp.matmul(intrinsics, camera_pose)
    return camera_matrix


def scale_intrinsics(camera_intrinsics, scale):
    focal_length_x = scale * camera_intrinsics[0, 0]
    focal_length_y = scale * camera_intrinsics[1, 1]
    center_x = scale * camera_intrinsics[0, 2]
    center_y = scale * camera_intrinsics[1, 2]
    scaled_intrinsics = jp.array(
        [
            [focal_length_x, 0.0, center_x],
            [0.0, focal_length_y, center_y],
            [0.0, 0.0, 1.0],
        ]
    )
    return scaled_intrinsics


def build_cube_corners(min_extents, max_extents):
    x_min, y_min, z_min = min_extents
    x_max, y_max, z_max = max_extents
    return jp.array(
        [
            [x_min, y_max, z_min],
            [x_max, y_max, z_min],
            [x_max, y_min, z_min],
            [x_min, y_min, z_min],
            [x_min, y_max, z_max],
            [x_max, y_max, z_max],
            [x_max, y_min, z_max],
            [x_min, y_min, z_max],
        ]
    )


def compute_AABB(vertices):
    min_extents = jp.min(vertices, axis=0)
    max_extents = jp.max(vertices, axis=0)
    return build_cube_corners(min_extents, max_extents)


def to_affine_matrix(rotation_matrix, translation):
    translation = translation.reshape(3, 1)
    affine_top = jp.concatenate([rotation_matrix, translation], axis=1)
    affine_row = jp.array([[0.0, 0.0, 0.0, 1.0]])
    affine_matrix = jp.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix


def compute_OBB(vertices_world):
    centroid_world = jp.mean(vertices_world, axis=0)
    centered_vertices_world = vertices_world - centroid_world
    centered_vertices_world_xy = centered_vertices_world[:, [0, 2]]
    u, s, vh = jp.linalg.svd(centered_vertices_world_xy)
    principal_axes = vh.T
    rx, ry, ux, uz = principal_axes.flatten()
    principal_axes = jp.array([[rx, 0.0, ry], [0.0, 1.0, 0], [ux, 0.0, uz]])
    centered_vertices_alpha = (principal_axes @ centered_vertices_world.T).T
    min_extents_alpha = jp.min(centered_vertices_alpha, axis=0)
    max_extents_alpha = jp.max(centered_vertices_alpha, axis=0)
    corners_world = build_cube_corners(min_extents_alpha, max_extents_alpha)
    world_to_alpha = to_affine_matrix(principal_axes, centroid_world)
    return (
        algebra.transform_points(world_to_alpha, corners_world),
        world_to_alpha,
    )


def project_box_and_vertices(intrinsics, world_to_camera, vertices_world):
    points3D, world_to_alpha = compute_OBB(vertices_world)
    camera_matrix = intrinsics @ world_to_camera
    points2D = vmap(partial(project_to_2D, camera_matrix))(points3D)
    vertices2D = vmap(partial(project_to_2D, camera_matrix))(vertices_world)
    return points2D.astype(int), vertices2D.astype(int)


def compute_focal_length_x(W, horizontal_field_of_view):
    return (W / 2) * (1 / jp.tan(jp.deg2rad(horizontal_field_of_view / 2.0)))


def intrinsics_from_HFOV(H, W, HFOV=70):
    """Computes camera intrinsics using horizontal field of view (HFOV).

    # Arguments
        HFOV: Angle in degrees of horizontal field of view.
        image_shape: List of two floats [H, W].

    # Returns
        camera intrinsics array (3, 3).

    # Notes:

                   \           /      ^
                    \         /       |
                     \ lens  /        | w/2
    horizontal field  \     / alpha/2 |
    of view (alpha)____\( )/_________ |      image
                       /( )\          |      plane
                      /     <-- f --> |
                     /       \        |
                    /         \       |
                   /           \      v

                Pinhole camera model

    From the image above we know that: tan(alpha/2) = w/2f
    -> f = w/2 * (1/tan(alpha/2))

    alpha in webcams and phones is often between 50 and 70 degrees.
    -> 0.7 w <= f <= w
    """
    focal_length = compute_focal_length(W, HFOV)
    camera_intrinsics = jp.array(
        [
            [focal_length, 0, W / 2.0],
            [0, focal_length, H / 2.0],
            [0, 0, 1.0],
        ]
    )


def calibrate(images, chessboard_size):
    H, W = chessboard_size
    object_points_2D = np.mgrid[0:W, 0:H].T.reshape(-1, 2)
    zeros = np.zeros((H * W, 1))
    object_points_3D = np.hstack((object_points_2D, zeros))
    object_points_3D = np.asarray(object_points_3D, dtype=np.float32)

    points2D, points3D = [], []
    for image in images:
        chessboard_found, corners = cv2.findChessboardCorners(image, (W, H))
        if chessboard_found:
            points2D.append(corners)
            points3D.append(object_points_3D)

    shape = image.shape[::-1]
    parameters = cv2.calibrateCamera(points3D, points2D, shape, None, None)
    _, camera_matrix, distortion_coefficient, _, _ = parameters
    return camera_matrix, distortion_coefficient
