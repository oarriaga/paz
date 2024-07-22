"""Predicting 3d poses from 2d joints"""
import os
import pickle
import numpy as np
from scipy.optimize import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from linear_model import mse_loss
import data_utils
import viz
import helper_functions

from tensorflow.keras.utils import get_file
from paz.applications import HigherHRNetHumanPose2D
from paz.backend.image import load_image, show_image
from paz.backend.camera import Camera

def joints_2d_from_image():
    # URL = ('https://github.com/oarriaga/altamira-data/releases/download'
    #        '/v0.10/single_person_test_pose.png')
    # filename = os.path.basename(URL)
    # fullpath = get_file(filename, URL, cache_subdir='paz/tests')
    path = '/home/dfki.uni-bremen.de/kshinde/Downloads/test_image.jpg'
    image = load_image(path)
    H, W = image.shape[:2]
    detect = HigherHRNetHumanPose2D()
    inferences = detect(image)
    # image = inferences['image']
    # show_image(image)
    return inferences['keypoints'], H, W


def optimize_trans(initial_root_translation, poses3d, Ki, f, img_center):
    """Optimization function to minimize the distance between 2d poses and projection of 3d poses

    Args
        initial_root_translation: initial guess of absolute position of root joint in camera space
        poses3d: 3d poses to be optimized
        Ki: 2d poses
        f: focal length
        img_center: principal point of the camera
    Returns
        person_sum: sum of L2 distances between each joint per person
    """
    # add root translation to poses3d
    initial_root_translation = np.reshape(initial_root_translation, (-1, 3))
    new_poses3d = poses3d + np.tile(initial_root_translation, (1, 16))

    # Project all poses translation 3D to 2D
    ppts = helper_functions.proj_3d_to_2d(new_poses3d.reshape((-1, 3)), f, img_center)
    ppts = ppts.reshape((Ki.shape[0],-1,2))
    Ki = Ki.reshape((Ki.shape[0],-1,2))
    person_sum = 0

    for i in range(Ki.shape[0]):
        person_sum += np.sum(np.linalg.norm(Ki[i] - ppts[i], axis=1))

    return person_sum


def predict_3d_poses():
    """Predicts 3d human pose for each person from the multi-human 2d poses obtained from HigherHRNet"""

    poses_2d, img_h, img_w = joints_2d_from_image()

    # Load 2d and 3d normalization stats for H36M dataset
    data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d, data_mean_3d, \
    data_std_3d, dim_to_use_3d, dim_to_ignore_3d = data_utils.load_params()
    print("\n==> done loading normalization stats.")

    poses_2d = data_utils.load_joints_2d(poses_2d)
    print(f"poses_2d : {poses_2d} {poses_2d.shape}")

    # Normalize 2d poses
    mu = data_mean_2d[dim_to_use_2d]
    stddev = data_std_2d[dim_to_use_2d]
    enc_in = np.divide((poses_2d - mu), stddev)

    # load the model
    model_path = '/home/dfki.uni-bremen.de/kshinde/Projects/models/baseline_model'
    # latter part added because custom loss is defined, is a TF bug
    model = tf.keras.models.load_model(model_path, custom_objects={'mse_loss': mse_loss})
    print("\n==> Model loaded!")

    # pass 2d poses and get predictions
    poses3d = model.predict(enc_in)

    # denormalize
    poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
    poses3d_copy = poses3d.copy()

    return poses_2d, poses3d, poses3d_copy, img_h, img_w


def translate_root():
    """Finds the optimal translation of root joint for each person to give a good enough estimate
    of the global human pose in camera coordinates"""

    poses_2d, poses3d, poses3d_copy, img_h, img_w = predict_3d_poses()

    p3d_16 = data_utils.filter_moving_joints_3d(poses3d)
    Ki = poses_2d.astype(np.float32) # 2d poses

    # get human root joint in 2d
    root_2d = poses_2d[:, :2]

    # FIXME: Get the intrinsics (image center and focal length) information from the image
    cam = Camera()
    cam.intrinsics_from_HFOV(HFOV=70, image_shape=[img_h, img_w])
    f = cam.intrinsics[0, 0]
    img_center = np.array([[cam.intrinsics[0, 2], cam.intrinsics[1, 2]]])

    s2d = helper_functions.s2d(poses_2d)
    s3d = helper_functions.s3d(p3d_16)

    initial_root_translation = helper_functions.init_translation(f, root_2d, img_center, s2d, s3d)
    initial_root_translation = initial_root_translation.flatten()

    root_translation = least_squares(optimize_trans, initial_root_translation, verbose=0, args=(p3d_16, Ki, f, img_center))

    print(f"\nOPTIMIZATION result : {root_translation}\n{root_translation.x}\n{root_translation.x.shape}")

    root_translation = np.reshape(root_translation.x, (-1, 3))
    root_translation = np.tile(root_translation, (1, 32))

    # Get global pose Pg = Pi + t*
    new_ppts = np.zeros(shape=(poses_2d.shape[0], 64))
    for i in range(poses3d.shape[0]):
        poses3d[i] = poses3d[i] + root_translation[i]
        ppts = helper_functions.proj_3d_to_2d(poses3d[i].reshape((-1, 3)), f, img_center)
        new_ppts[i] = np.reshape(ppts, [1, 64])
    print(f"\nposes3d after optimization {poses3d} {poses3d.shape}")
    print(f"\nRoots after optimization {poses3d[:,:3]} {poses3d[:,:3].shape}")

    visualize(poses_2d, p3d_16, poses3d, new_ppts)


def visualize(poses_2d, poses3d, ps3d, ppts):
    # 1080p	= 1,920 x 1,080
    fig = plt.figure(figsize=(19.2, 10.8))
    gs1 = gridspec.GridSpec(1,4)
    gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    plt.axis('off')

    ax = plt.subplot(gs1[0])
    viz.show2Dpose(poses_2d, ax, add_labels=True)
    ax.invert_yaxis()
    ax.title.set_text('HRNet 2D poses')

    ax1 = plt.subplot(gs1[1], projection='3d')
    ax1.view_init(-90, -90)
    viz.show3Dpose(poses3d, ax1, add_labels=True)
    ax1.title.set_text('Baseline prediction')

    ax2 = plt.subplot(gs1[2], projection='3d')
    ax2.view_init(-90, -90)
    viz.show3Dpose(ps3d, ax2, add_labels=True)
    ax2.title.set_text('Optimized 3D poses')

    ax3 = plt.subplot(gs1[3])
    viz.show2Dpose(ppts, ax3, add_labels=True)
    ax3.invert_yaxis()
    ax3.title.set_text('2D projection of optimized poses')
    plt.show()


if __name__ == "__main__":
    translate_root()
