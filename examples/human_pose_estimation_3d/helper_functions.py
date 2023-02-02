import numpy as np
from paz.backend.camera import Camera
from scipy.optimize import least_squares


def project_3D_to_2D(points_3D_wrt_camera, focal_length, image_center):
    """ Project points in camera frame from 3D to 2D using intrinsic matrix 
    # Arguments
        points_3D_wrt_camera: Nx3 points in camera coordinates
        focal_length: (scalar) Camera focal length
        img_center: 2x1 image center      
    # Returns
        Nx2 points in pixel space
    """
    z = points_3D_wrt_camera[:, 2]
    u = (focal_length / z) * points_3D_wrt_camera[:, 0] + image_center[0, 0]
    v = (focal_length / z) * points_3D_wrt_camera[:, 1] + image_center[0, 1]
    return np.column_stack((u, v))


def get_bones_length(poses2D, poses3D):
    """Computes sum of bone lengths in 3D

    # Arguments
        poses3D: np array of predicted poses in 3D
        poses2D: np array of poses in 2D

    # Returns
        sum_bones2D: sum of length of all bones in the 3D skeleton
        sum_bones3D: sum of length of all bones in the 3D skeleton
    """
    poses2D = np.reshape(poses2D, (len(poses2D), 16, -1))
    poses3D = np.reshape(poses3D, (len(poses3D), 16, -1))
    sum_bones2D = np.zeros(poses2D.shape[0])
    sum_bones3D = np.zeros(poses3D.shape[0])
    start_joints = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    end_joints = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    for idx, person in enumerate(poses2D):
        for joint in np.arange(len(start_joints)):
            bone_length = np.linalg.norm(
                person[start_joints[joint]] - person[end_joints[joint]])
            sum_bones2D += bone_length
    for idx, person in enumerate(poses3D):
        for joint in np.arange(len(start_joints)):
            bone_length = np.linalg.norm(
                person[start_joints[joint]] - person[end_joints[joint]])
            sum_bones3D += bone_length
    return sum_bones2D, sum_bones3D


def initialize_translation(focal_length, joints2D, image_center, ratio):
    """Computes initial 3D translation of root joint

    # Arguments
        focal_length: focal length of the camera in pixels
        joints_2D: 2D root joint from HigherHRNet
        image_center: center of the image (or principal point)
        sum_bones2D: sum of bone lengths of 2D skeleton
        sum_bones3D: sum of bone lengths of 3D skeleton 

    # Returns
        Array of initial estimate of the global position
        of the root joint in 3D
    """
    camera_center_X, camera_center_Y = image_center
    translation_Z = focal_length * ratio  # depth coord
    joints2D_X, joints2D_Y = np.split(joints2D, 2, axis=1)
    joints2D_X = np.reshape(joints2D_X, (len(joints2D_X),))
    joints2D_Y = np.reshape(joints2D_Y, (len(joints2D_Y),))
    translation_X = (joints2D_X - camera_center_X) * ratio  # horz coord
    translation_Y = (joints2D_Y - camera_center_Y) * ratio  # vert coord
    translation = np.column_stack(
        (translation_X, translation_Y, translation_Z))
    return translation.flatten()


def get_cam_parameters(image_height, image_width):
    """Computes orthographic projection of 3D pose

    # Arguments
        image_heigth: height of image
        image_width: width of image

    # Returns
        focal length and image center
    """
    camera = Camera()
    camera.intrinsics_from_HFOV(HFOV=70,
                                image_shape=[image_height, image_width])
    focal_length = camera.intrinsics[0, 0]
    image_center = np.array(
        [[camera.intrinsics[0, 2], camera.intrinsics[1, 2]]])
    return focal_length, image_center


def solve_least_squares(compute_joints_distance, initial_joints_translation,
                        moving_joints_3D, poses2D, focal_length, image_center):
    """ Solve the least squares    
    # Arguments
        compute_joints_distance: global_pose.compute_joints_distance
        initial_root_translation: initial 3D translation of root joint
        moving_joints_3D: 16 moving joints in 3D
        poses2d: 2D poses
        focal_length: focal length
        img_center: image center

    Returns
        optimal translation of root joint for each person
    """
    joints_translation = least_squares(compute_joints_distance,
                                       initial_joints_translation, verbose=0,
                                       args=(
                                           moving_joints_3D, poses2D,
                                           focal_length,
                                           image_center))
    joints_translation = np.reshape(joints_translation.x, (-1, 3))
    joints_translation = np.tile(joints_translation, (1, 32))
    return joints_translation
