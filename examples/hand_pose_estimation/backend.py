import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

from hand_keypoints_loader import kinematic_chain_dict, kinematic_chain_list
from hand_keypoints_loader import LEFT_ROOT_KEYPOINT_ID
from hand_keypoints_loader import LEFT_ALIGNED_KEYPOINT_ID
from hand_keypoints_loader import LEFT_LAST_KEYPOINT_ID

from hand_keypoints_loader import RIGHT_ROOT_KEYPOINT_ID
from hand_keypoints_loader import RIGHT_ALIGNED_KEYPOINT_ID
from hand_keypoints_loader import RIGHT_LAST_KEYPOINT_ID


def extract_hand_segment(segmentation_label):
    hand_mask = np.greater(segmentation_label, 1)
    background_mask = np.logical_not(hand_mask)
    return np.stack([background_mask, hand_mask], 2)


def transform_visibility_mask(visibility_mask):
    palm_visibility_left = np.logical_or(
        visibility_mask[LEFT_ROOT_KEYPOINT_ID],
        visibility_mask[LEFT_ALIGNED_KEYPOINT_ID])
    palm_visibility_left = np.expand_dims(palm_visibility_left, 0)
    palm_visibility_right = np.logical_or(
        visibility_mask[RIGHT_ROOT_KEYPOINT_ID],
        visibility_mask[RIGHT_ALIGNED_KEYPOINT_ID])
    palm_visibility_right = np.expand_dims(palm_visibility_right, 0)
    visibility_mask = np.concatenate([palm_visibility_left,
                                      visibility_mask[1:21],
                                      palm_visibility_right,
                                      visibility_mask[21:43]], 0)
    return visibility_mask


def keypoints_to_wrist_coordinates(keypoints):
    palm_coordinates_left = np.expand_dims(
        0.5 * (keypoints[LEFT_ROOT_KEYPOINT_ID, :] +
               keypoints[LEFT_ALIGNED_KEYPOINT_ID, :]), 0)
    palm_coordinates_right = np.expand_dims(
        0.5 * (keypoints[RIGHT_ROOT_KEYPOINT_ID, :] +
               keypoints[RIGHT_ALIGNED_KEYPOINT_ID, :]), 0)
    keypoints = np.concatenate(
        [palm_coordinates_left,
         keypoints[LEFT_ROOT_KEYPOINT_ID + 1:LEFT_LAST_KEYPOINT_ID + 1, :],
         palm_coordinates_right,
         keypoints[RIGHT_ROOT_KEYPOINT_ID + 1:RIGHT_LAST_KEYPOINT_ID + 1, :]],
        0)
    return keypoints


def one_hot_encode(inputs, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector
    for each label.
    : input: List of sample Labels
    : return: Numpy array of one-hot encoded labels """
    return np.eye(n_classes, dtype=int)[inputs]


def normalize_keypoints(keypoints3D):
    keypoint3D_root = keypoints3D[0, :]
    relative_keypoint3D = keypoints3D - keypoint3D_root
    keypoint_scale = np.linalg.norm(
        relative_keypoint3D[LEFT_ALIGNED_KEYPOINT_ID, :] -
        relative_keypoint3D[LEFT_ALIGNED_KEYPOINT_ID - 1, :])

    keypoint_normalized = relative_keypoint3D / keypoint_scale
    return keypoint_scale, keypoint_normalized


def to_homogeneous_coordinates(vector):
    vector = np.append(vector, 1)
    return vector


def build_4D_translation_matrix(translation_vector):
    transformation_matrix = np.array([[1, 0, 0, translation_vector[0]],
                                      [0, 1, 0, translation_vector[1]],
                                      [0, 0, 1, translation_vector[2]],
                                      [0, 0, 0, 1]])
    return transformation_matrix


def get_translation_matrix(translation_vector):
    if len(translation_vector) == 1:
        transformation_matrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, translation_vector],
                                          [0, 0, 0, 1]])
    else:
        transformation_matrix = build_4D_translation_matrix(translation_vector)

    transformation_matrix = np.expand_dims(transformation_matrix, 0)
    return transformation_matrix


def get_affine_matrix(matrix):
    t = np.array([0, 0, 0]).reshape(3, 1)  # rename
    affine_matrix = np.hstack([matrix, t])
    affine_matrix = np.vstack((affine_matrix, [0, 0, 0, 1]))
    return affine_matrix


def build_rotation_matrix_x(angle):
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle), np.sin(angle)],
                                  [0, -np.sin(angle), np.cos(angle)]])
    return rotation_matrix_x


def build_rotation_matrix_y(angle):
    rotation_matrix_y = np.array([[np.cos(angle), 0, -np.sin(angle)],
                                  [0, 1, 0],
                                  [np.sin(angle), 0, np.cos(angle)]])
    return rotation_matrix_y


def build_rotation_matrix_z(angle):
    rotation_matrix_z = np.array([[np.cos(angle), np.sin(angle), 0],
                                  [-np.sin(angle), np.cos(angle), 0],
                                  [0, 0, 1]])
    return rotation_matrix_z


def extract_hand_masks(hand_parts_mask, right_hand_mask_limit=17):
    one_map = np.ones_like(hand_parts_mask)

    mask_left = np.logical_and(np.greater(hand_parts_mask, one_map),
                               np.less(hand_parts_mask, one_map *
                                       (right_hand_mask_limit + 1)))
    mask_right = np.greater(hand_parts_mask, one_map * right_hand_mask_limit)

    hand_mask_left = mask_left.astype('int')
    hand_mask_right = mask_right.astype('int')

    return hand_mask_left, hand_mask_right


def extract_dominant_hand_mask(keypoints3D, dominant_hand):  # Function name
    keypoint3D_left = keypoints3D[
                      LEFT_ROOT_KEYPOINT_ID:LEFT_LAST_KEYPOINT_ID, :]
    keypoints_mask = np.ones_like(keypoint3D_left, dtype=bool)
    dominant_hand_mask = np.logical_and(keypoints_mask, dominant_hand)
    return dominant_hand_mask


def extract_hand_side_keypooints(keypoints3D, dominant_hand_mask):
    keypoint3D_left = keypoints3D[
                      LEFT_ROOT_KEYPOINT_ID:LEFT_LAST_KEYPOINT_ID + 1, :]
    keypoint3D_right = keypoints3D[
                       RIGHT_ROOT_KEYPOINT_ID:RIGHT_LAST_KEYPOINT_ID + 1, :]
    hand_side_keypoints = np.where(dominant_hand_mask, keypoint3D_left,
                                   keypoint3D_right)
    return hand_side_keypoints


def get_hand_side_and_keypooints(hand_parts_mask, keypoints3D):
    hand_map_left, hand_map_right = extract_hand_masks(hand_parts_mask)

    num_pixels_hand_left = np.sum(hand_map_left)
    num_pixels_hand_right = np.sum(hand_map_right)

    dominant_hand = np.greater(num_pixels_hand_left, num_pixels_hand_right)

    dominant_hand_mask = extract_dominant_hand_mask(keypoints3D, dominant_hand)
    hand_side_keypoints = extract_hand_side_keypooints(keypoints3D,
                                                       dominant_hand_mask)

    hand_side = np.where(dominant_hand, 0, 1)

    return hand_side, hand_side_keypoints, dominant_hand_mask


def transform_to_relative_frame(keypoints_3D, bone_index):
    translated_keypoint3D = to_homogeneous_coordinates(
        np.expand_dims(keypoints_3D[bone_index, :], 1))

    Translation_matrix = get_translation_matrix(
        np.zeros_like(keypoints_3D[0, 0]))

    transformation_parameters = get_transformation_parameters(
        translated_keypoint3D, Translation_matrix)
    return transformation_parameters


def get_local_coordinates(transformation_matrix, keypoint3D):
    homogeneous_keypoint3D = to_homogeneous_coordinates(
        np.expand_dims(keypoint3D, 1))
    local_keypoint_coordinates = np.matmul(transformation_matrix,
                                           homogeneous_keypoint3D)
    return local_keypoint_coordinates


def get_root_transformations(keypoints_3D, bone_index,
                             relative_coordinates, transformations):
    transformation_parameters = transform_to_relative_frame(
        keypoints_3D, bone_index)
    relative_coordinates[bone_index] = np.stack(
        transformation_parameters[:3], 0)
    transformations[bone_index] = transformation_parameters[3]
    return transformations, relative_coordinates


def get_articulation_angles(local_child_coordinates, local_parent_coordinates,
                            Transformation_matrix):
    delta_vector = local_child_coordinates - local_parent_coordinates
    delta_vector = to_homogeneous_coordinates(np.expand_dims(
        delta_vector[:, :3], 1))

    transformation_parameters = transform_to_relative_frame(
        delta_vector, Transformation_matrix)
    return transformation_parameters


def get_child_transformations(keypoints_3D, bone_index, parent_key,
                              relative_coordinates, transformations):
    Transformation_matrix = transformations[parent_key]

    local_parent_coordinates = get_local_coordinates(
        Transformation_matrix, keypoints_3D[parent_key, :])
    local_child_coordinates = get_local_coordinates(
        Transformation_matrix, keypoints_3D[bone_index, :])

    transformation_parameters = get_articulation_angles(
        local_child_coordinates, local_parent_coordinates,
        Transformation_matrix)

    relative_coordinates[bone_index] = np.stack(
        transformation_parameters[:3])
    transformations[bone_index] = transformation_parameters[3]
    return transformations, relative_coordinates


def get_keypoints_relative_frame(keypoints_3D):
    transformations = [None] * len(kinematic_chain_list)
    relative_coordinates = [0.0] * len(kinematic_chain_list)

    for bone_index in kinematic_chain_list:
        parent_key = kinematic_chain_dict[bone_index]
        if parent_key == 'root':
            transformations, relative_coordinates = get_root_transformations(
                keypoints_3D, bone_index, relative_coordinates, transformations)
        else:
            transformations, relative_coordinates = get_child_transformations(
                keypoints_3D, bone_index, parent_key, relative_coordinates,
                transformations)

    return relative_coordinates


def transform_to_relative_frames(keypoints_3D):
    keypoints_3D = keypoints_3D.reshape([21, 3])

    relative_coordinates = get_keypoints_relative_frame(keypoints_3D)

    key_point_relative_frame = np.stack(relative_coordinates, 1)

    return key_point_relative_frame


def get_keypoints_z_rotation(alignment_keypoint, translated_keypoints3D):
    alpha = np.arctan2(alignment_keypoint[0], alignment_keypoint[1])
    rotation_matrix_z = build_rotation_matrix_z(alpha)
    resultant_keypoints3D = np.matmul(translated_keypoints3D.T,
                                      rotation_matrix_z)

    reference_keypoint_z_rotation = resultant_keypoints3D[
                                    LEFT_ALIGNED_KEYPOINT_ID, :]
    return reference_keypoint_z_rotation, resultant_keypoints3D, \
           rotation_matrix_z


def get_keypoints_x_rotation(keypoints3D, rotation_matrix):
    beta = -np.arctan2(rotation_matrix[2], rotation_matrix[1])
    rotation_matrix_x = build_rotation_matrix_x(beta + np.pi)

    resultant_keypoints3D = np.matmul(keypoints3D, rotation_matrix_x)

    resultant_keypoint = resultant_keypoints3D[LEFT_LAST_KEYPOINT_ID, :]
    return resultant_keypoint, rotation_matrix_x, resultant_keypoints3D


def get_keypoints_y_rotation(keypoints3D, reference_keypoint_z_rotation):
    gamma = np.arctan2(reference_keypoint_z_rotation[2],
                       reference_keypoint_z_rotation[0])
    rotation_matrix_y = build_rotation_matrix_y(gamma)

    transformed_keypoints3D = np.matmul(keypoints3D, rotation_matrix_y)
    return transformed_keypoints3D, rotation_matrix_y


def get_canonical_transformations(keypoints3D):
    reference_keypoint = np.expand_dims(keypoints3D[:, LEFT_ROOT_KEYPOINT_ID],
                                        1)
    translated_keypoints3D = keypoints3D - reference_keypoint
    alignment_keypoint = translated_keypoints3D[:, LEFT_ALIGNED_KEYPOINT_ID]

    reference_keypoint_z_rotation, resultant_keypoints3D, rotation_matrix_z = \
        get_keypoints_z_rotation(alignment_keypoint, translated_keypoints3D)

    reference_keypoint_x_rotation, rotation_matrix_x, resultant_keypoints3D \
        = get_keypoints_x_rotation(resultant_keypoints3D,
                                   reference_keypoint_z_rotation)

    transformed_keypoints3D, rotation_matrix_y = get_keypoints_y_rotation(
        resultant_keypoints3D, reference_keypoint_x_rotation
    )

    final_rotation_matrix = np.matmul(np.matmul(rotation_matrix_z,
                                                rotation_matrix_x),
                                      rotation_matrix_y)
    return np.squeeze(transformed_keypoints3D), \
           np.squeeze(final_rotation_matrix)


def get_best_crop_size(max_coordinates, min_coordinates, crop_center):
    crop_size_best = 2 * np.maximum(max_coordinates - crop_center,
                                    crop_center - min_coordinates)
    crop_size_best = np.amax(crop_size_best)
    crop_size_best = np.minimum(np.maximum(crop_size_best, 50.0), 500.0)
    if not np.isfinite(crop_size_best):
        crop_size_best = 200.0
    return crop_size_best


def get_scale_matrix(scale):
    scale_original = np.minimum(np.maximum(scale, 1.0), 10.0)

    scale = np.reshape(scale_original, [1, ])
    scale_matrix = np.array([scale, 0.0, 0.0,
                             0.0, scale, 0.0,
                             0.0, 0.0, 1.0])
    scale_matrix = np.reshape(scale_matrix, [3, 3])
    return scale_original, scale_matrix


def get_scale_translation_matrix(crop_center, crop_size, scale):
    trans1 = crop_center[0] * scale - crop_size // 2
    trans2 = crop_center[1] * scale - crop_size // 2

    trans1 = np.reshape(trans1, [1, ])
    trans2 = np.reshape(trans2, [1, ])

    trans_matrix = np.array([1.0, 0.0, -trans2,
                             0.0, 1.0, -trans1,
                             0.0, 0.0, 1.0])

    trans_matrix = np.reshape(trans_matrix, [3, 3])
    return trans_matrix


def extract_coordinate_limits(keypoints_2D, keypoints_2D_vis, image_size):
    keypoint_h = keypoints_2D[:, 1][keypoints_2D_vis]
    keypoint_w = keypoints_2D[:, 0][keypoints_2D_vis]
    kp_coord_hw = np.stack([keypoint_h, keypoint_w], 1)

    min_coordinates = np.maximum(np.amin(kp_coord_hw, 0), 0.0)
    max_coordinates = np.minimum(np.amax(kp_coord_hw, 0), image_size[0:2])
    return min_coordinates, max_coordinates


def get_keypoints_camera_coordinates(keypoints_2D, crop_center, scale,
                                     crop_size):
    keypoint_uv21_u = (keypoints_2D[:, 0] - crop_center[1]) * scale + \
                      crop_size // 2
    keypoint_uv21_v = (keypoints_2D[:, 1] - crop_center[0]) * scale + \
                      crop_size // 2
    keypoint_uv21 = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
    return keypoint_uv21


def get_scale(keypoints_2D, keypoints_2D_vis, image_size, crop_size):
    crop_center = keypoints_2D[12, ::-1]
    crop_center = np.reshape(crop_center, [2, ])
    min_coordinates, max_coordinates = extract_coordinate_limits(
        keypoints_2D, keypoints_2D_vis, image_size)

    crop_size_best = get_best_crop_size(max_coordinates, min_coordinates,
                                        crop_center)

    scale = crop_size / crop_size_best
    return scale, crop_center


def crop_image_using_mask(keypoints_2D, keypoints_2D_vis, image, image_size,
                          crop_size, camera_matrix):
    scale, crop_center = get_scale(keypoints_2D, keypoints_2D_vis, image_size,
                                   crop_size)
    scale, scale_matrix = get_scale_matrix(scale)

    img_crop = crop_image_from_coordinates(image, crop_center, crop_size, scale)

    keypoint_uv21 = get_keypoints_camera_coordinates(keypoints_2D, crop_center,
                                                     scale, crop_size)

    scale_translation_matrix = get_scale_translation_matrix(crop_center,
                                                            crop_size, scale)

    camera_matrix_cropped = np.matmul(scale_translation_matrix,
                                      np.matmul(scale_matrix, camera_matrix))

    return scale, np.squeeze(img_crop), keypoint_uv21, camera_matrix_cropped


def flip_right_hand(canonical_keypoints3D, flip_right):
    shape = canonical_keypoints3D.shape
    expanded = False
    if len(shape) == 2:
        canonical_keypoints3D = np.expand_dims(canonical_keypoints3D, 0)
        flip_right = np.expand_dims(flip_right, 0)
        expanded = True
    canonical_keypoints3D_mirrored = np.stack(
        [canonical_keypoints3D[:, :, 0], canonical_keypoints3D[:, :, 1],
         -canonical_keypoints3D[:, :, 2]], -1)

    canonical_keypoints3D_left = np.where(flip_right,
                                          canonical_keypoints3D_mirrored,
                                          canonical_keypoints3D)
    if expanded:
        canonical_keypoints3D_left = np.squeeze(canonical_keypoints3D_left,
                                                axis=0)
    return canonical_keypoints3D_left


def extract_dominant_hand_visibility(keypoint_visibility, dominant_hand):
    keypoint_visibility_left = keypoint_visibility[:21]
    keypoint_visibility_right = keypoint_visibility[-21:]
    keypoint_visibility_21 = np.where(dominant_hand[:, 0],
                                      keypoint_visibility_left,
                                      keypoint_visibility_right)
    return keypoint_visibility_21


def extract_dominant_keypoints2D(keypoint_2D, dominant_hand):
    keypoint_visibility_left = keypoint_2D[:21, :]
    keypoint_visibility_right = keypoint_2D[-21:, :]
    keypoint_visibility_2D_21 = np.where(dominant_hand[:, :2],
                                         keypoint_visibility_left,
                                         keypoint_visibility_right)
    return keypoint_visibility_2D_21


def extract_keypoint2D_limits(uv_coordinates, scoremap_size):
    cond_1_in = np.logical_and(np.less(uv_coordinates[:, 0],
                                       scoremap_size[0] - 1),
                               np.greater(uv_coordinates[:, 0], 0))
    cond_2_in = np.logical_and(np.less(uv_coordinates[:, 1],
                                       scoremap_size[1] - 1),
                               np.greater(uv_coordinates[:, 1], 0))

    cond_in = np.logical_and(cond_1_in, cond_2_in)

    return cond_in


def get_keypoints_mask(valid_vec, uv_coordinates, scoremap_size):
    if valid_vec is not None:
        valid_vec = np.squeeze(valid_vec)
        keypoint_validity = np.greater(valid_vec, 0.5)
    else:
        keypoint_validity = np.ones_like(uv_coordinates[:, 0], dtype=np.float32)
        keypoint_validity = np.greater(keypoint_validity, 0.5)

    keypoint_out_limits = extract_keypoint2D_limits(uv_coordinates,
                                                    scoremap_size)
    keypooints_mask = np.logical_and(keypoint_validity, keypoint_out_limits)
    return keypooints_mask


def get_xy_limits(uv_coordinates, scoremap_size):
    shape = uv_coordinates.shape
    x_range = np.expand_dims(np.arange(scoremap_size[0]), 1)
    y_range = np.expand_dims(np.arange(scoremap_size[1]), 0)

    X = np.tile(x_range, [1, scoremap_size[1]])
    Y = np.tile(y_range, [scoremap_size[0], 1])

    X.reshape((scoremap_size[0], scoremap_size[1]))
    Y.reshape((scoremap_size[0], scoremap_size[1]))

    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)

    X_coordinates = np.tile(X, [1, 1, shape[0]])
    Y_coordinates = np.tile(Y, [1, 1, shape[0]])

    X_limits = X_coordinates - uv_coordinates[:, 0].astype('float64')
    Y_limits = Y_coordinates - uv_coordinates[:, 1].astype('float64')

    return X_limits, Y_limits


def create_multiple_gaussian_map(uv_coordinates, scoremap_size, sigma,
                                 valid_vec):
    assert len(scoremap_size) == 2
    keypoints_mask = get_keypoints_mask(valid_vec, uv_coordinates,
                                        scoremap_size)

    X_limits, Y_limits = get_xy_limits(uv_coordinates, scoremap_size)

    dist = np.square(X_limits) + np.square(Y_limits)

    scoremap = np.exp(-dist / np.square(sigma)) * keypoints_mask

    return scoremap


def get_transformation_parameters(keypoint3D, transformation_matrix):
    length_from_origin = np.linalg.norm(keypoint3D)

    gamma = np.arctan2(keypoint3D[0], keypoint3D[2])
    rotation_matrix_y = build_rotation_matrix_y(gamma)
    affine_rotation_matrix_y = get_affine_matrix(rotation_matrix_y)

    keypoint3D_rotated_Y = np.matmul(affine_rotation_matrix_y, keypoint3D)

    alpha = np.arctan2(-keypoint3D_rotated_Y[1], keypoint3D_rotated_Y[2])
    rotation_matrix_x = build_rotation_matrix_x(alpha)
    affine_rotation_matrix_x = get_affine_matrix(rotation_matrix_x)

    translation_matrix_to_origin = get_translation_matrix(-length_from_origin)
    rotation_matrix_xy = np.matmul(affine_rotation_matrix_x,
                                   affine_rotation_matrix_y)

    keypoint3D_rotated_X = np.matmul(translation_matrix_to_origin,
                                     rotation_matrix_xy)

    final_transformation_matrix = np.matmul(keypoint3D_rotated_X,
                                            transformation_matrix)

    return length_from_origin, alpha, gamma, final_transformation_matrix


def get_XY_arrays(shape):
    x_range = np.expand_dims(np.arange(shape[1]), 1)
    y_range = np.expand_dims(np.arange(shape[2]), 0)

    X = tf.tile(x_range, [1, shape[2]])
    Y = tf.tile(y_range, [shape[1], 1])
    return X, Y


def get_bounding_box_list(X_masked, Y_masked, bounding_box_list):
    x_min, x_max, y_min, y_max = np.min(X_masked), np.max(X_masked), \
                                 np.min(Y_masked), np.max(Y_masked)

    xy_limits = [x_min, x_max, y_min, y_max]
    start = np.stack([x_min, y_min])
    end = np.stack([x_max, y_max])
    bounding_box = np.stack([start, end], 1)
    bounding_box_list.append(bounding_box)
    return bounding_box_list, xy_limits


def get_center_list(xy_limit, center_list):
    center_x = 0.5 * (xy_limit[1] + xy_limit[0])
    center_y = 0.5 * (xy_limit[3] + xy_limit[2])

    center = np.stack([center_x, center_y], 0)

    if not np.all(np.isfinite(center)):
        center = np.array([160, 160])
    center.reshape([2])
    center_list.append(center)
    return center_list


def get_crop_list(xy_limit, crop_size_list):
    crop_size_x = xy_limit[1] - xy_limit[0]
    crop_size_y = xy_limit[3] - xy_limit[2]
    crop_maximum_value = np.maximum(crop_size_x, crop_size_y)
    crop_size = np.expand_dims(crop_maximum_value, 0)
    crop_size.reshape([1])
    crop_size_list.append(crop_size)
    return crop_size_list


def get_bounding_box_features(X, Y, binary_class_mask, shape):
    bounding_box_list, center_list, crop_size_list = list()
    for i in range(shape[0]):
        X_masked = X[binary_class_mask[i, :, :]].numpy().astype(np.float)
        Y_masked = Y[binary_class_mask[i, :, :]].numpy().astype(np.float)

        bounding_box_list, xy_limit = get_bounding_box_list(X_masked, Y_masked,
                                                            bounding_box_list)

        center_list = get_center_list(xy_limit, center_list)

        crop_size_list = get_crop_list(xy_limit, crop_size_list)
    return bounding_box_list, center_list, crop_size_list


def extract_bounding_box(binary_class_mask):
    binary_class_mask = binary_class_mask.numpy().astype(np.int)
    binary_class_mask = np.equal(binary_class_mask, 1)
    shape = binary_class_mask.shape

    assert len(shape) == 3, "binary_class_mask must be 3D."

    X, Y = get_XY_arrays(shape)

    bounding_box_list, center_list, crop_size_list = get_bounding_box_features(
        X, Y, binary_class_mask, shape)

    bounding_box = np.stack(bounding_box_list)
    center = np.stack(center_list)
    crop_size = np.stack(crop_size_list)

    return center, bounding_box, crop_size


def convert_location_to_box(location, size, shape):
    y1 = location[:, 0] - size // 2
    y2 = y1 + size
    x1 = location[:, 1] - size // 2
    x2 = x1 + size
    y1 /= shape[1]
    y2 /= shape[1]
    x1 /= shape[2]
    x2 /= shape[2]
    boxes = np.stack([y1, x1, y2, x2], -1)
    return boxes


def crop_image_from_coordinates(image, crop_location, crop_size, scale=1.0):
    image_dimensions = image.shape
    scale = np.reshape(scale, [-1])
    crop_location = crop_location.astype(np.float)
    crop_location = np.reshape(crop_location, [image_dimensions[0], 2])
    crop_size = float(crop_size)

    crop_size_scaled = crop_size / scale

    boxes = convert_location_to_box(crop_location, crop_size_scaled,
                                    image_dimensions)

    crop_size = np.stack([crop_size, crop_size])
    crop_size = crop_size.astype(np.float)
    box_indices = np.arange(image_dimensions[0])
    image_cropped = tf.image.crop_and_resize(tf.cast(image, tf.float32),
                                             boxes, box_indices, crop_size,
                                             name='crop')
    return image_cropped.numpy()


def find_max_location(scoremap):
    """ Returns the coordinates of the given scoremap with maximum value. """

    s = scoremap.shape
    assert len(s) == 3, "Scoremap must be 3D."

    x_range = np.expand_dims(np.arange(s[1]), 1)
    y_range = np.expand_dims(np.arange(s[2]), 0)
    X = np.tile(x_range, [1, s[2]])
    Y = np.tile(y_range, [s[1], 1])

    x_vec = np.reshape(X, [-1])
    y_vec = np.reshape(Y, [-1])
    scoremap_vec = np.reshape(scoremap, [s[0], -1])
    max_ind_vec = np.argmax(scoremap_vec, axis=1)
    max_ind_vec = max_ind_vec.astype(np.int)

    xy_loc = list()
    for i in range(s[0]):
        x_loc = np.reshape(x_vec[max_ind_vec[i]], [1])
        y_loc = np.reshape(y_vec[max_ind_vec[i]], [1])
        xy_loc.append(np.concatenate([x_loc, y_loc], 0))

    xy_loc = np.stack(xy_loc, 0)
    return xy_loc


def object_scoremap(scoremap):
    filter_size = 21
    s = scoremap.shape
    assert len(s) == 4, "Scoremap must be 4D."

    scoremap_softmax = tf.nn.softmax(scoremap)
    scoremap_fg = tf.reduce_max(scoremap_softmax[:, :, :, 1:], -1)  # B, H, W
    detmap_fg = tf.round(scoremap_fg)  # B, H, W

    max_loc = find_max_location(scoremap_fg)

    objectmap_list = list()
    kernel_dil = tf.ones((filter_size, filter_size, 1)) / float(
        filter_size * filter_size)
    if s[0] is None:
        s[0] = 1
    for i in range(s[0]):
        sparse_ind = tf.reshape(max_loc[i, :], [1, 2])
        objectmap = tf.compat.v1.sparse_to_dense(sparse_ind, [s[1], s[2]], 1.0)

        num_passes = max(s[1], s[2]) // (filter_size // 2)
        for j in range(num_passes):
            objectmap = tf.reshape(objectmap, [1, s[1], s[2], 1])
            objectmap_dil = tf.compat.v1.nn.dilation2d(objectmap, kernel_dil,
                                                       [1, 1, 1, 1],
                                                       [1, 1, 1, 1], 'SAME')

            objectmap_dil = tf.reshape(objectmap_dil, [s[1], s[2]])
            objectmap = tf.round(tf.multiply(detmap_fg[i, :, :], objectmap_dil))

        objectmap = tf.reshape(objectmap, [s[1], s[2], 1])
        objectmap_list.append(objectmap)

    objectmap = tf.stack(objectmap_list)

    return objectmap


def get_rotation_matrix(rot_params):
    theta = np.linalg.norm(rot_params)

    sine_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    one_ct = 1.0 - cos_theta

    normalization_factor = 1.0 / theta
    ux = rot_params[:, 0] * normalization_factor
    uy = rot_params[:, 1] * normalization_factor
    uz = rot_params[:, 2] * normalization_factor

    row_1 = np.stack((cos_theta + ux * ux * one_ct,
                      ux * uy * (1.0 - cos_theta) - uz * sine_theta,
                      ux * uz * (1.0 - cos_theta) + uy * sine_theta), axis=1)
    row_2 = np.stack((uy * ux * (1.0 - cos_theta) + uz * sine_theta,
                      cos_theta + uy * uy * (1.0 - cos_theta),
                      uy * uz * (1.0 - cos_theta) - ux * sine_theta), axis=1)
    row_3 = np.stack((uz * ux * (1.0 - cos_theta) - uy * sine_theta,
                      uz * uy * (1.0 - cos_theta) + ux * sine_theta,
                      cos_theta + uz * uz * one_ct), axis=1)

    rot_matrix = np.stack((row_1, row_2, row_3), axis=1)
    return rot_matrix


def create_score_maps(keypoint_2D, keypoint_vis21, image_size, crop_size,
                      sigma, crop_image=True):
    keypoint_hw21 = np.stack([keypoint_2D[:, 1], keypoint_2D[:, 0]], -1)

    scoremap_size = image_size[0:2]

    if crop_image:
        scoremap_size = (crop_size, crop_size)

    scoremap = create_multiple_gaussian_map(keypoint_hw21,
                                            scoremap_size,
                                            sigma,
                                            valid_vec=keypoint_vis21)

    return scoremap


def extract_2D_keypoints(visibility_mask):
    keypoints2D = visibility_mask[:, :2]
    keypoints_visibility_mask = visibility_mask[:, 2] == 1
    return keypoints2D, keypoints_visibility_mask


def detect_keypoints(scoremaps):
    """ Performs detection per scoremap for the hands keypoints. """
    scoremaps_shape = scoremaps.shape

    keypoint_coords = np.zeros((scoremaps_shape[2], 2))
    for i in range(scoremaps_shape[2]):
        v, u = np.unravel_index(np.argmax(scoremaps[:, :, i]),
                                (scoremaps_shape[0], scoremaps_shape[1]))
        keypoint_coords[i, 0] = v
        keypoint_coords[i, 1] = u
    return keypoint_coords


def wrap_dictionary(keys, values):
    return dict(zip(keys, values))


def merge_dictionaries(dicts):
    result = {}
    for dict in dicts:
        result.update(dict)
    return result
