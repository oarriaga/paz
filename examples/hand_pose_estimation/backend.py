import numpy as np

from paz.backend.image.opencv_image import resize_image, crop_image


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


def extract_hand_side(hand_parts_mask, key_point_3D):
    one_map, zero_map = np.ones_like(hand_parts_mask), np.zeros_like(
        hand_parts_mask)

    raw_mask_left = np.logical_and(np.greater(hand_parts_mask, one_map),
                                   np.less(hand_parts_mask, one_map * 18))
    raw_mask_right = np.greater(hand_parts_mask, one_map * 17)

    hand_map_left = np.where(raw_mask_left, one_map, zero_map)
    hand_map_right = np.where(raw_mask_right, one_map, zero_map)

    num_pixels_left_hand = np.sum(hand_map_left)
    num_pixels_right_hand = np.sum(hand_map_right)

    kp_coord_xyz_left = key_point_3D[0:21, :]
    kp_coord_xyz_right = key_point_3D[21:43, :]

    dominant_hand = np.logical_and(np.ones_like(kp_coord_xyz_left,
                                                dtype=bool),
                                   np.greater(num_pixels_left_hand,
                                              num_pixels_right_hand))

    hand_side_keypoints = np.where(dominant_hand, kp_coord_xyz_left,
                                   kp_coord_xyz_right)

    hand_side = np.where(np.greater(num_pixels_left_hand,
                                    num_pixels_right_hand), 0, 1)

    return hand_side, hand_side_keypoints, dominant_hand


def get_canonical_transformations(keypoints_3D):
    keypoints = np.reshape(keypoints_3D, [21, 3])

    ROOT_NODE_ID = 0
    ALIGN_NODE_ID = 12
    LAST_NODE_ID = 20

    translation_reference = np.expand_dims(keypoints[ROOT_NODE_ID, :], 1)
    translated_keypoints = keypoints - translation_reference.T

    alignment_keypoint = translated_keypoints[ALIGN_NODE_ID, :]

    alpha = np.arctan2(alignment_keypoint[0], alignment_keypoint[1])
    rotation_matrix_z = get_rotation_matrix_z(alpha)
    resultant_keypoints = np.matmul(translated_keypoints, rotation_matrix_z)

    reference_keypoint_z_rotation = resultant_keypoints[ALIGN_NODE_ID, :]
    beta = -np.arctan2(reference_keypoint_z_rotation[2],
                       reference_keypoint_z_rotation[1])
    rotation_matrix_x = get_rotation_matrix_x(beta + 3.14159)
    resultant_keypoints = np.matmul(resultant_keypoints, rotation_matrix_x)

    reference_keypoint_z_rotation = resultant_keypoints[LAST_NODE_ID, :]
    gamma = np.arctan2(reference_keypoint_z_rotation[2],
                       reference_keypoint_z_rotation[0])
    rotation_matrix_y = get_rotation_matrix_y(gamma)
    keypoints_transformed = np.matmul(resultant_keypoints,
                                      rotation_matrix_y)

    final_rotation_matrix = np.matmul(np.matmul(rotation_matrix_z,
                                                rotation_matrix_x),
                                      rotation_matrix_y)
    return np.squeeze(keypoints_transformed), \
           np.squeeze(final_rotation_matrix)


def flip_right_hand(canonical_keypoints, flip_right):
    shape = canonical_keypoints.shape
    expanded = False
    if len(shape) == 2:
        canonical_keypoints = np.expand_dims(canonical_keypoints, 0)
        flip_right = np.expand_dims(flip_right, 0)
        expanded = True
    canonical_keypoints_mirrored = np.stack(
        [canonical_keypoints[:, :, 0], canonical_keypoints[:, :, 1],
         -canonical_keypoints[:, :, 2]], -1)

    canonical_keypoints_left = np.where(flip_right,
                                        canonical_keypoints_mirrored,
                                        canonical_keypoints)
    if expanded:
        canonical_keypoints_left = np.squeeze(canonical_keypoints_left,
                                              axis=0)
    return canonical_keypoints_left


def extract_dominant_hand_visibility(keypoint_visibility, dominant_hand):
    keypoint_visibility_left = keypoint_visibility[:21]
    keypoint_visibility_right = keypoint_visibility[-21:]
    keypoint_visibility_21 = np.where(dominant_hand[:, 0],
                                      keypoint_visibility_left,
                                      keypoint_visibility_right)
    return keypoint_visibility_21


def extract_dominant_2D_keypoints(keypoint_2D_visibility, dominant_hand):
    keypoint_visibility_left = keypoint_2D_visibility[:21, :]
    keypoint_visibility_right = keypoint_2D_visibility[-21:, :]
    keypoint_visibility_2D_21 = np.where(dominant_hand[:, :2],
                                         keypoint_visibility_left,
                                         keypoint_visibility_right)
    return keypoint_visibility_2D_21


def crop_image_from_coordinates(image, crop_location, crop_size,
                                scale=0.1):
    crop_size_scaled = crop_size / scale

    y1 = crop_location[0] - crop_size_scaled // 2
    y2 = crop_location[0] + crop_size_scaled // 2
    x1 = crop_location[1] - crop_size_scaled // 2
    x2 = crop_location[1] + crop_size_scaled // 2

    box = [int(x1), int(y1), int(x2), int(y2)]
    box = [max(min(x, 320), 0) for x in box]
    print(box)

    final_image_size = (crop_size, crop_size)
    image_cropped = crop_image(image, box)

    image_resized = resize_image(image_cropped, final_image_size)
    return image_resized


def create_multiple_gaussian_map(uv_coordinates, scoremap_size, sigma,
                                 valid_vec):
    assert len(scoremap_size) == 2
    s = uv_coordinates.shape
    coords_uv = uv_coordinates
    if valid_vec is not None:
        valid_vec = np.squeeze(valid_vec)
        cond_val = np.greater(valid_vec, 0.5)
    else:
        cond_val = np.ones_like(coords_uv[:, 0], dtype=np.float32)
        cond_val = np.greater(cond_val, 0.5)

    cond_1_in = np.logical_and(np.less(coords_uv[:, 0],
                                       scoremap_size[0] - 1),
                               np.greater(coords_uv[:, 0], 0))
    cond_2_in = np.logical_and(np.less(coords_uv[:, 1],
                                       scoremap_size[1] - 1),
                               np.greater(coords_uv[:, 1], 0))
    cond_in = np.logical_and(cond_1_in, cond_2_in)
    cond = np.logical_and(cond_val, cond_in)

    # create meshgrid
    x_range = np.expand_dims(np.arange(scoremap_size[0]), 1)
    y_range = np.expand_dims(np.arange(scoremap_size[1]), 0)

    X = np.tile(x_range, [1, scoremap_size[1]])
    Y = np.tile(y_range, [scoremap_size[0], 1])

    X.reshape((scoremap_size[0], scoremap_size[1]))
    Y.reshape((scoremap_size[0], scoremap_size[1]))

    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)

    X_b = np.tile(X, [1, 1, s[0]])
    Y_b = np.tile(Y, [1, 1, s[0]])

    X_b = X_b - coords_uv[:, 0].astype('float64')
    Y_b = Y_b - coords_uv[:, 1].astype('float64')

    dist = np.square(X_b) + np.square(Y_b)

    scoremap = np.exp(-dist / np.square(sigma)) * cond

    return scoremap


def get_geometric_entities(vector, transformation_matrix):
    length_from_origin = np.linalg.norm(vector)
    gamma = np.arctan2(vector[0, 0], vector[2, 0])

    matrix_after_y_rotation = np.matmul(
        get_transformation_matrix_y(-gamma), vector)
    alpha = np.arctan2(-matrix_after_y_rotation[1, 0],
                       matrix_after_y_rotation[2, 0])
    matrix_after_x_rotation = np.matmul(get_translation_matrix(
        -length_from_origin), np.matmul(get_transformation_matrix_x(
        -alpha), get_transformation_matrix_y(-gamma)))

    final_transformation_matrix = np.matmul(matrix_after_x_rotation,
                                            transformation_matrix)

    # make them all batched scalars
    length_from_origin = np.reshape(length_from_origin, [-1])
    alpha = np.reshape(alpha, [-1])
    gamma = np.reshape(gamma, [-1])
    return length_from_origin, alpha, gamma, final_transformation_matrix
