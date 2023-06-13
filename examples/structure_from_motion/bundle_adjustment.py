import numpy as np
from scipy.optimize import least_squares
from paz.backend.keypoints import project_to_image
from paz.backend.groups import rotation_matrix_to_compact_axis_angle
from paz.backend.groups import rotation_vector_to_rotation_matrix


def residuals(camera_pose, points3D, points2D, camera_intrinsics):
    rotation = camera_pose[:3]
    rotation = rotation_vector_to_rotation_matrix(rotation)
    translation = camera_pose[3: 6]
    project2D = project_to_image(rotation, translation, points3D,
                                 camera_intrinsics)
    joints_distance = np.linalg.norm(points2D - project2D, axis=1)
    return joints_distance


def local_bundle_adjustment(rotation, translation, points3D, points2D,
                            camera_intrinsics):
    num_points = points3D.shape[0]
    axis_angle = rotation_matrix_to_compact_axis_angle(rotation)
    camera_pose = np.concatenate([axis_angle, translation.reshape(-1)])
    param_init = np.hstack((camera_pose, points3D.ravel()))

    result = least_squares(residuals, param_init,
                           args=(points3D, points2D, camera_intrinsics))

    optimized_params = result.x

    # Extract the optimized camera poses and 3D points
    optimized_camera_poses = optimized_params[:6]
    optimized_point_cloud = optimized_params[6:].reshape((num_points, 3))

    return optimized_point_cloud, optimized_camera_poses
