import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from paz.backend.quaternion import quarternion_to_rotation_matrix


def evaluateMSPD(real_box_pixels, predicted_box_pixels, img_size):
    # Normalize the pixels
    real_box_pixels[:, 0] /= img_size[0]
    real_box_pixels[:, 1] /= img_size[1]

    predicted_box_pixels[:, 0] /= img_size[0]
    predicted_box_pixels[:, 1] /= img_size[1]

    distances = list()
    for real_point, predicted_point in zip(real_box_pixels, predicted_box_pixels):
        distances.append(np.linalg.norm(real_point - predicted_point))

    distances = np.asarray(distances)
    return np.max(distances)


def evaluateMSSD(real_box, predicted_box):
    distances = list()
    for real_point, predicted_point in zip(real_box, predicted_box):
        distances.append(np.linalg.norm(real_point - predicted_point))

    distances = np.asarray(distances)
    return np.max(distances)


def evaluateADD(real_box, predicted_box):
    distances = list()
    for real_point, predicted_point in zip(real_box, predicted_box):
        distances.append(np.linalg.norm(real_point - predicted_point))

    distances = np.asarray(distances)
    return np.mean(distances)


def evaluateIoU(real_box_pose, predicted_box_pose, box_extents, num_sampled_points=1000):
    """Calculates the 3D intersection over union between 'real_box' and all 'predicted_box'.
    Both `box` and `boxes` are in corner coordinates.
    # Arguments
        box: Numpy array with length at least of 4.
        boxes: Numpy array with shape `(num_boxes, 4)`.
    # Returns
        Numpy array of shape `(num_boxes, 1)`.
    """
    # Sample values between -1 and 1
    sampled_points = np.random.uniform(low=-1., high=1., size=(num_sampled_points, 3))
    # Scale it to the box extents
    sampled_points = np.multiply(sampled_points, box_extents)
    # Rotate and move the points (real pose)
    sampled_points_rotated = np.asarray([quarternion_to_rotation_matrix(real_box_pose.quaternion) @ bb_point + np.squeeze(real_box_pose.translation) for bb_point in sampled_points])
    # Rotate and move the points back (predicted pose)
    sampled_points_back_rotated = np.asarray([quarternion_to_rotation_matrix(predicted_box_pose.quaternion).T @ (bb_point - np.squeeze(predicted_box_pose.translation)) for bb_point in sampled_points_rotated])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sampled_points_back_rotated[:, 0], sampled_points_back_rotated[:, 1], sampled_points_back_rotated[:, 2])
    plt.show()

    num_points_inside = 0
    for i, sampled_point_rotated in enumerate(sampled_points_back_rotated):
        if -box_extents[0] <= sampled_point_rotated[0] <= box_extents[0] and \
           -box_extents[1] <= sampled_point_rotated[1] <= box_extents[1] and \
           -box_extents[2] <= sampled_point_rotated[2] <= box_extents[2]:

            num_points_inside += 1

    return num_points_inside/num_sampled_points