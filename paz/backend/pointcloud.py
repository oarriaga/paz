import jax.numpy as jp

from paz.backend import algebra


def split(pointcloud):
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]
    return x, y, z


def bound(pointcloud, max_depth):
    z = pointcloud[:, 2]
    is_close_by = z < max_depth
    is_not_zero = jp.logical_not(jp.isclose(z, 0.0))
    mask = jp.logical_and(is_close_by, is_not_zero)
    return pointcloud[mask.flatten().astype(bool)]


def merge(list_of_pointclouds):
    return jp.concatenate(list_of_pointclouds, axis=0)


def transform(pointcloud, affine_transform):
    return algebra.transform_points(affine_transform, pointcloud)


def color(pointcloud, color=[0, 255, 0]):
    return jp.repeat(jp.array([color]), len(pointcloud), axis=0)


def mask(pointcloud, mask, max_depth):
    z = pointcloud[:, 2]
    is_close_by = z < max_depth
    is_not_zero = jp.logical_not(jp.isclose(z, 0.0))
    is_shape = mask.flatten().astype(bool)
    mask = jp.logical_and(is_shape, jp.logical_and(is_close_by, is_not_zero))
    return pointcloud[mask.flatten().astype(bool)]


def from_depth(depth, camera_intrinsics):
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    u_center = camera_intrinsics[0, 2]
    v_center = camera_intrinsics[1, 2]
    H, W = depth.shape[:2]
    u_options = jp.arange(0, W)
    v_options = jp.arange(0, H)
    u_grid, v_grid = jp.meshgrid(u_options, v_options)
    u_indices = u_grid.flatten()
    v_indices = v_grid.flatten()
    depths = depth.flatten()
    x = depths * (u_indices - u_center) / fx
    y = depths * (v_indices - v_center) / fy
    z = depths
    return jp.vstack([x, y, z]).T


def to_depth(pointcloud, camera_intrinsics, H, W, min_depth=0.0):
    x, y, z = split(pointcloud)
    # Filter out points behind the camera
    valid_mask = z > 0
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    # project points to 2D image coordinates
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    u = ((fx * x / z) + cx).astype(int)
    v = ((fy * y / z) + cy).astype(int)
    # initialize depth image
    depth_image = jp.full((H, W), jp.inf)
    # Keep points within image bounds
    valid_u = jp.logical_and(u >= 0, u < W)
    valid_v = jp.logical_and(v >= 0, v < H)
    valid_pixels = jp.logical_and(valid_u, valid_v)
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    for point_arg in range(len(u)):
        x_pixel_arg = u[point_arg]
        y_pixel_arg = v[point_arg]
        z_value_new = z[point_arg]
        z_value_old = depth_image[y_pixel_arg, x_pixel_arg]
        if (min_depth < z_value_new < z_value_old):
            depth_image[y_pixel_arg, x_pixel_arg] = z_value_new
    depth_image = jp.where(depth_image == jp.inf, 0, depth_image)
    return depth_image


def sample(key, pointcloud, num_points):
    point_args = jp.random.choice(
        key, len(pointcloud), shape=(num_points,), replace=False)
    return pointcloud[point_args]


def to_camera_coordinates(pointcloud, camera_intrinsics):
    x, y, z = split(pointcloud)
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    u_center = camera_intrinsics[0, 2]
    v_center = camera_intrinsics[1, 2]
    u = (fx * x / z) + u_center
    v = (fy * y / z) + v_center
    return jp.vstack([u, v]).T


def remove_outliers(pointcloud, num_stdvs=3):
    data_mean = jp.mean(pointcloud, axis=0)
    data_stdv = jp.std(pointcloud, axis=0)
    cut_off = data_stdv * num_stdvs
    lower_cutoff = data_mean - cut_off
    upper_cutoff = data_mean + cut_off
    is_above_lower_cutoff = jp.all(pointcloud > lower_cutoff, axis=1)
    is_below_upper_cutoff = jp.all(pointcloud < upper_cutoff, axis=1)
    mask = jp.logical_or(is_above_lower_cutoff, is_below_upper_cutoff)
    return pointcloud[mask]
