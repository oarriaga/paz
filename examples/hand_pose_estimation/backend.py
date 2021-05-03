import numpy as np
import cv2


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    res = res.astype(int)
    return res.reshape(list(targets.shape) + [nb_classes])


def normalize_keypoints(keypoints):
    keypoint_coord_xyz_root = keypoints[0, :]
    keypoint_coord_xyz21_rel = keypoints - keypoint_coord_xyz_root
    keypoint_scale = np.sqrt(np.sum(
        np.square(keypoint_coord_xyz21_rel[12, :] -
                  keypoint_coord_xyz21_rel[11, :])))
    keypoint_normalized = keypoint_coord_xyz21_rel / keypoint_scale
    return keypoint_scale, keypoint_normalized


def to_homogeneous_coordinates(vector):
    vector = np.append(vector, 1)
    vector = np.reshape(vector, [-1, 1])
    return vector


def get_translation_matrix(translation_vector):
    transformation_vector = np.array([1, 0, 0, 0,
                                      0, 1, 0, 0,
                                      0, 0, 1, translation_vector,
                                      0, 0, 0, 1])
    transformation_matrix = np.reshape(transformation_vector, [4, 4, 1])
    transformation_matrix = np.transpose(transformation_matrix, [2, 0, 1])

    return transformation_matrix


def get_transformation_matrix_x(angle):
    rotation_matrix_x = np.array([1, 0, 0, 0,
                                  0, np.cos(angle), -np.sin(angle), 0,
                                  0, np.sin(angle), np.cos(angle), 0,
                                  0, 0, 0, 1])
    transformation_matrix = np.reshape(rotation_matrix_x, [4, 4, 1])
    transformation_matrix = np.transpose(transformation_matrix, [2, 0, 1])
    return transformation_matrix


def get_transformation_matrix_y(angle):
    rotation_matrix_y = np.array([np.cos(angle), 0, np.sin(angle), 0,
                                  0, 1, 0, 0,
                                  -np.sin(angle), 0, np.cos(angle), 0,
                                  0, 0, 0, 1])
    transformation_matrix = np.reshape(rotation_matrix_y, [4, 4])
    # transformation_matrix = np.transpose(transformation_matrix, [2, 0, 1])
    return transformation_matrix


def get_rotation_matrix_x(angle):
    rotation_matrix_x = np.array([1, 0, 0,
                                  0, np.cos(angle), np.sin(angle),
                                  0, -np.sin(angle), np.cos(angle)])
    rotation_matrix = np.reshape(rotation_matrix_x, [3, 3])
    return rotation_matrix


def get_rotation_matrix_y(angle):
    rotation_matrix_y = np.array([np.cos(angle), 0, -np.sin(angle),
                                  0, 1, 0,
                                  np.sin(angle), 0, np.cos(angle)])
    rotation_matrix = np.reshape(rotation_matrix_y, [3, 3])
    return rotation_matrix


def get_rotation_matrix_z(angle):
    rotation_matrix_z = np.array([np.cos(angle), np.sin(angle), 0,
                                  -np.sin(angle), np.cos(angle), 0,
                                  0, 0, 1])
    rotation_matrix = np.reshape(rotation_matrix_z, [3, 3])
    return rotation_matrix


def crop_resize_image(image, box, new_size):
    crooped_image = image[box[1]:box[3], box[0]:box[2]]
