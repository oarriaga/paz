"""Predicting 3d poses from 2d joints"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import viz
from backend import standardize
from helper_functions import initialize_translation, project_3D_to_2D, \
    get_cam_parameters, get_bones_length, solve_least_squares
from data_utils import unnormalize_data, filter_keypoints_3D, load_joints_2D
from paz.applications import HigherHRNetHumanPose2D
from paz.backend.image import load_image
from linear_model import SIMPLE_BASELINE
from data import mean2D, stdd2D, data_mean3D, \
        data_std3D, dim_to_use3D


def joints_2D_from_image():
    path = 'test_image.jpg'
    image = load_image(path)
    H, W = image.shape[:2]
    detect = HigherHRNetHumanPose2D()
    inferences = detect(image)
    return inferences['keypoints'], H, W


def compute_joints_distance(initial_joint_translation, poses3D, poses2D,
                            focal_length, img_center):
    """compute distance etween each person joints

    # Arguments
        initial_joint_translation: initial guess of position of joint
        poses3d: 3D poses to be optimized
        poses2d: 2D poses
        focal_length: focal length
        img_center: principal point of the camera

    # Returns
        person_sum: sum of L2 distances between each joint per person
    """
    initial_joint_translation = np.reshape(initial_joint_translation, (-1, 3))
    new_poses3D = poses3D + np.tile(initial_joint_translation, (1, 16))
    proj_2D = project_3D_to_2D(new_poses3D.reshape((-1, 3)), focal_length,
                               img_center)
    proj_2D = proj_2D.reshape((poses2D.shape[0], -1, 2))
    poses2D = poses2D.reshape((poses2D.shape[0], -1, 2))
    joints_distance =  [np.linalg.norm(poses2D[i] - proj_2D[i], axis=1)
                        for i in range(len(poses2D))]
    return np.sum(joints_distance)


def predict_3d_keypoints():
    """Predicts 3d human pose for each person from the m
    ulti-human 2D poses obtained from HigherHRNet"""

    poses2D, image_h, image_w = joints_2D_from_image()
    poses2D = load_joints_2D(poses2D)
    norm_data = standardize(poses2D, mean2D, stdd2D)
    model = SIMPLE_BASELINE(1024, (32,), 2, True, True, True, 1)
    model.load_weights('weights.h5')
    poses3D = model.predict(norm_data)
    poses3D = unnormalize_data(poses3D, data_mean3D, data_std3D,
                               dim_to_use3D)
    return poses2D, poses3D, image_h, image_w


def solve_translation():
    """Finds the optimal translation of root joint for each person
       to give a good enough estimate of the global human pose
       in camera coordinates"""

    poses2D, poses3D, image_h, image_w = predict_3d_keypoints()
    joints_3D = filter_keypoints_3D(poses3D)
    root_2D = poses2D[:, :2]
    focal_length, image_center = get_cam_parameters(image_h, image_w)
    length_2D, length_3D = get_bones_length(poses2D, joints_3D)
    ratio = length_3D / length_2D
    initial_joint_translation = initialize_translation(focal_length,
                                                       root_2D,
                                                       image_center[0],
                                                       ratio)
    joint_translation = solve_least_squares(compute_joints_distance,
                                            initial_joint_translation,
                                            joints_3D, poses2D,
                                            focal_length, image_center)
    new_points = np.zeros(shape=(poses2D.shape[0], 64))
    for i in range(poses3D.shape[0]):
        poses3D[i] = poses3D[i] + joint_translation[i]
        points = project_3D_to_2D(poses3D[i].reshape((-1, 3)), focal_length,
                                  image_center)
        new_points[i] = np.reshape(points, [1, 64])
    visualize(poses2D, joints_3D, poses3D, new_points)


def visualize(poses2D, poses3D, points_3D, opimized_pose_3D):
    """Vizualize points

    # Arguments
        poses2D: 2D poses
        poses3D: 3D poses
        points_3D:
        opimized_pose_3D:
    """
    fig = plt.figure(figsize=(19.2, 10.8))
    gs1 = gridspec.GridSpec(1, 4)
    gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    plt.axis('off')
    ax = plt.subplot(gs1[0])
    viz.show2Dpose(poses2D, ax, add_labels=True)
    ax.invert_yaxis()
    ax.title.set_text('HRNet 2D poses')
    ax1 = plt.subplot(gs1[1], projection='3d')
    ax1.view_init(-90, -90)
    viz.show3Dpose(poses3D, ax1, add_labels=True)
    ax1.title.set_text('Baseline prediction')
    ax2 = plt.subplot(gs1[2], projection='3d')
    ax2.view_init(-90, -90)
    viz.show3Dpose(points_3D, ax2, add_labels=True)
    ax2.title.set_text('Optimized 3D poses')
    ax3 = plt.subplot(gs1[3])
    viz.show2Dpose(opimized_pose_3D, ax3, add_labels=True)
    ax3.invert_yaxis()
    ax3.title.set_text('2D projection of optimized poses')
    plt.show()


if __name__ == "__main__":
    solve_translation()
