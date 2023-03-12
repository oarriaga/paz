import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import viz
import human36m


def compute_reprojection_error(initial_translation, keypoints3D,
                               keypoints2D, focal_length, image_center):
    """compute distance between each person joints
    # Arguments
        initial_translation: initial guess of position of joint
        keypoints3D: 3D keypoints to be optimized (Nx16x3)
        keypoints2D: 2D keypoints (Nx32)
        focal_length: focal length
        img_center: principal point of the camera
    # Returns
        person_sum: sum of L2 distances between each joint per person
    """
    initial_translation = np.reshape(initial_translation, (-1, 3))
    new_poses3D = np.zeros((keypoints3D.shape))
    for person in range(len(initial_translation)):
        new_poses3D[person] = keypoints3D[person] + initial_translation[person]
    project2D = project_3D_to_2D(new_poses3D.reshape((-1, 3)), focal_length,
                                 image_center)
    joints_distance = np.linalg.norm(np.ravel(keypoints2D) -
                                     np.ravel(project2D))
    return np.sum(joints_distance)


def solve_translation3D(keypoints2D, keypoints3D, focal_length, image_center,
                        args_to_joints3D):
    """Finds the optimal translation of root joint for each person
    to give a good enough estimate of the global human pose
    in camera coordinates
    #Arguments
        keypoints2D: array of keypoints in 2D (Nx32)
        keypoints3D: array of keypoints in 3D (Nx96)
        image_height: height of the image
        image_width: width of image
    #Returns
        keypoints2D: array of keypoints in 2D
        Joints3D: array of joints in 3D
        keypoints3D: array of keypoints in 3D
        optimezed_poses3D: optimized pose 3D
    """
    joints3D = human36m.filter_keypoints3D(keypoints3D, args_to_joints3D)
    root2D = keypoints2D[:, :2]
    length2D, length3D = get_bones_length(keypoints2D, joints3D)
    ratio = length3D / length2D
    initial_joint_translation = initialize_translation(focal_length, root2D,
                                                       image_center, ratio)
    joint_translation = solve_least_squares(compute_reprojection_error,
                                            initial_joint_translation,
                                            joints3D, keypoints2D,
                                            focal_length, image_center)
    optimized_poses3D = []
    keypoints3D = np.reshape(keypoints3D, (-1, 32, 3))
    for person in range(keypoints3D.shape[0]):
        keypoints3D[person] = keypoints3D[person] + joint_translation[person]
        points = project_3D_to_2D(keypoints3D[person].reshape((-1, 3)),
                                  focal_length, image_center)
        optimized_poses3D.append(np.reshape(points, [1, 64]))
    return joints3D, keypoints3D, np.array(optimized_poses3D)


def project_3D_to_2D(points3D, focal_length, image_center):
    """ Project points in camera frame from 3D to 2D using intrinsic matrix
    # Arguments
        points3D: Nx3 points in camera coordinates (32x3)
        focal_length: (scalar) Camera focal length
        img_center: 2x1 image center
    # Returns
        Nx2 points in pixel space
    """
    z = points3D[:, 2]
    u = (focal_length / z) * points3D[:, 0] + image_center[0]
    v = (focal_length / z) * points3D[:, 1] + image_center[1]
    return np.column_stack((u, v))


def get_bones_length(poses2D, poses3D):
    """Computes sum of bone lengths in 3D
    #Arguments
        poses3D: array of predicted poses in 3D (Nx16x3)
        poses2D: array of poses in 2D    (Nx32)
    #Returns
        sum_bones2D: sum of length of all bones in the 3D skeleton
        sum_bones3D: sum of length of all bones in the 3D skeleton
    """
    sum_bones2D = 0
    sum_bones3D = np.zeros(poses3D.shape[0])
    start_joints = np.arange(0, 15)
    end_joints = np.arange(1, 16)
    for person in poses2D:
        bone_length = np.linalg.norm(person[start_joints] - person[end_joints])
        sum_bones2D += bone_length
    for person in poses3D:
        bone_length = np.linalg.norm(person[start_joints] - person[end_joints])
        sum_bones3D += bone_length
    return sum_bones2D, sum_bones3D


def initialize_translation(focal_length, joints2D, image_center, ratio):
    """Computes initial 3D translation of root joint
    # Arguments
        focal_length: focal length of the camera in pixels
        joints2D: 2D root joint from HigherHRNet
        image_center: center of the image (or principal point)
        sum_bones2D: sum of bone lengths of 2D skeleton
        sum_bones3D: sum of bone lengths of 3D skeleton
    # Returns
        Array of initial estimate of the global position
        of the root joint in 3D
    """
    z = focal_length * ratio  # depth coord
    x = (joints2D[:, 0] - image_center[0]) * ratio
    y = (joints2D[:, 1] - image_center[1]) * ratio
    translation = np.array((x, y, z))
    return translation.flatten()


def solve_least_squares(compute_joints_distance, initial_joints_translation,
                        joints3D, poses2D, focal_length, image_center):
    """ Solve the least squares
    # Arguments
        compute_joints_distance: global_pose.compute_joints_distance
        initial_root_translation: initial 3D translation of root joint
        joints3D: 16 moving joints in 3D
        poses2d: 2D poses
        focal_length: focal length
        img_center: image center
    Returns
        optimal translation of root joint for each person
    """
    joints_translation = least_squares(
        compute_joints_distance, initial_joints_translation, verbose=0,
        args=(joints3D, poses2D, focal_length, image_center))
    joints_translation = np.reshape(joints_translation.x, (-1, 3))
    return joints_translation


def visualize(keypoints2D, joints3D, keypoints3D, opimized_pose3D):
    """Vizualize points
    # Arguments
        keypoints2D: 2D poses
        joints3D: 3D poses
        keypoints3D:
        opimized_pose_3D: Optimized posed3D
    """
    plt.figure(figsize=(19.2, 10.8))
    grid_spec = gridspec.GridSpec(1, 4)
    grid_spec.update(wspace=-0.00, hspace=0.05)
    plt.axis('off')
    axis1 = plt.subplot(grid_spec[0])
    viz.show2Dpose(keypoints2D, axis1, add_labels=True)
    axis1.invert_yaxis()
    axis1.title.set_text('HRNet 2D poses')
    axis2 = plt.subplot(grid_spec[1], projection='3d')
    axis2.view_init(-90, -90)
    viz.show3Dpose(joints3D, axis2, add_labels=True)
    axis2.title.set_text('Baseline prediction')
    axis3 = plt.subplot(grid_spec[2], projection='3d')
    axis3.view_init(-90, -90)
    viz.show3Dpose(keypoints3D, axis3, add_labels=True)
    axis3.title.set_text('Optimized 3D poses')
    axis4 = plt.subplot(grid_spec[3])
    viz.show2Dpose(opimized_pose3D, axis4, add_labels=True)
    axis4.invert_yaxis()
    axis4.title.set_text('2D projection of optimized poses')
    plt.show()


def standardize(data, mean, scale):
    """Standardize the data.
    # Arguments
        data: nxd matrix to normalize
        mean: Array of means
        scale: standard deviation
    # Returns
        standardized poses2D
    # """
    return np.divide((data - mean), scale)


def destandardize(data, mean, scale):
    """Destandardize the data.
    # Arguments
        data: nxd matrix to unnormalize
        mean: Array of means
        scale: standard deviation
    # Returns
        destandardized poses3D
    """
    return (data * scale) + mean
