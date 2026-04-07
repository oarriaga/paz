import jax
import jax.numpy as jp
import numpy as np
import paz


def intersect_line_and_plane(point2D, camera_intrinsics, theta_plane):
    u, v = point2D
    pixel = jp.array([u, v, 1.0])
    x, y, z = jp.linalg.inv(camera_intrinsics) @ pixel
    a, b, d = theta_plane
    line_scale = d / ((a * x) + (b * y) - z)
    return line_scale * jp.array([x, y, z])


def fit_RANSAC(seed, pointcloud, threshold=0.01):
    key = jax.random.PRNGKey(seed)
    pointcloud = jp.array(pointcloud)
    result = paz.plane.fit_RANSAC(key, pointcloud, threshold=threshold)
    normal, offset, inlier_mask = result
    normal = np.array(normal)
    offset = float(offset)
    z_scale = normal[2] if abs(normal[2]) > 1e-8 else -1e-8
    values = [-normal[0] / z_scale, -normal[1] / z_scale, -offset / z_scale]
    plane = np.array(values)
    return plane, np.array(inlier_mask)


def build_z_plane_in_camera(plane):
    a, b, _ = plane
    normal = jp.array([a, b, -1.0])
    return normal / jp.linalg.norm(normal)


def build_plane_origin_in_camera(H, W, camera_intrinsics, plane):
    intersect = paz.lock(intersect_line_and_plane, camera_intrinsics, plane)
    center_point2D = jp.array([W / 2.0, H / 2.0])
    return -intersect(center_point2D)


def build_x_plane_in_camera(H, W, camera_intrinsics, plane):
    intersect = paz.lock(intersect_line_and_plane, camera_intrinsics, plane)
    center_args = (H, W, camera_intrinsics, plane)
    center_point3D = build_plane_origin_in_camera(*center_args)
    lower_point2D = jp.array([W / 2.0, H])
    lower_point3D = -intersect(lower_point2D)
    x_axis = lower_point3D - center_point3D
    return x_axis / jp.linalg.norm(x_axis)


def build_plane_pose(H, W, camera_intrinsics, plane):
    z_axis = build_z_plane_in_camera(plane)
    x_axis = build_x_plane_in_camera(H, W, camera_intrinsics, plane)
    y_axis = np.cross(np.array(z_axis), np.array(x_axis))
    y_axis = y_axis / np.linalg.norm(y_axis)
    rotation = jp.column_stack([x_axis, jp.array(y_axis), z_axis])
    position = build_plane_origin_in_camera(H, W, camera_intrinsics, plane)
    return paz.SE3.to_affine_matrix(rotation, position)


def fit_camera_to_plane(seed, pointcloud, image_size, camera_intrinsics):
    H, W = image_size
    plane_camera_openCV, _ = fit_RANSAC(seed, pointcloud)
    pose_args = (H, W, camera_intrinsics, plane_camera_openCV)
    plane_to_camera_openCV = build_plane_pose(*pose_args)
    rot = paz.SE3.rotation_x(jp.pi / 2.0)
    return jp.linalg.inv(plane_to_camera_openCV @ rot)


