import numpy as np


def normalize_keypoints(keypoints):
    keypoint_coord_xyz_root = keypoints[0, :]
    keypoint_coord_xyz21_rel = keypoints - keypoint_coord_xyz_root
    index_root_bone_length = np.linalg.norm(keypoint_coord_xyz21_rel[12, :] -
                                            keypoint_coord_xyz21_rel[11, :])
    keypoint_scale = index_root_bone_length
    keypoint_normalized = keypoint_coord_xyz21_rel / index_root_bone_length
    return keypoint_scale, keypoint_normalized


def to_homogeneous_coordinates(vector):
    vector = np.reshape(vector, [1, -1, 1])
    vector = np.concat([vector, np.ones((1, 1, 1))], 1)
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


def get_rotation_matrix_z(self, angle):
    rotation_matrix_z = np.array([np.cos(angle), np.sin(angle), 0,
                                  -np.sin(angle), np.cos(angle), 0,
                                  0, 0, 1])
    rotation_matrix = np.reshape(rotation_matrix_z, [3, 3])
    return rotation_matrix
