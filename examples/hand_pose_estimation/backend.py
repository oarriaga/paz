import numpy as np
import tensorflow as tf

from paz.backend.image.opencv_image import resize_image, crop_image


def one_hot_encode(input, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector
    for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[input]


def normalize_keypoints(keypoints3D):
    keypoint3D_root = keypoints3D[0, :]
    relative_keypoint3D = keypoints3D - keypoint3D_root
    keypoint_scale = np.linalg.norm(relative_keypoint3D[12, :] -
                                    relative_keypoint3D[11, :])

    keypoint_normalized = relative_keypoint3D / keypoint_scale
    return keypoint_scale, keypoint_normalized


def to_homogeneous_coordinates(vector):
    vector = np.append(vector, 1)
    return vector


def get_translation_matrix(translation_vector):
    transformation_matrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, translation_vector],
                                      [0, 0, 0, 1]])
    transformation_matrix = np.expand_dims(transformation_matrix, 0)

    return transformation_matrix


def get_affine_matrix(matrix):
    t = np.array([0, 0, 0]).reshape(3, 1)
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


def extract_hand_side(hand_parts_mask, keypoints3D):
    one_map = np.ones_like(hand_parts_mask)

    RIGHT_HAND_MASK_LIMITS = 17

    mask_left = np.logical_and(np.greater(hand_parts_mask, one_map),
                               np.less(hand_parts_mask, one_map *
                                       (RIGHT_HAND_MASK_LIMITS + 1)))
    mask_right = np.greater(hand_parts_mask, one_map * RIGHT_HAND_MASK_LIMITS)

    hand_map_left = mask_left.astype('int')
    hand_map_right = mask_right.astype('int')

    num_pixels_hand_left = np.sum(hand_map_left)
    num_pixels_hand_right = np.sum(hand_map_right)

    keypoint3D_left = keypoints3D[0:21, :]
    keypoint3D_right = keypoints3D[21:43, :]

    dominant_hand = np.greater(num_pixels_hand_left, num_pixels_hand_right)
    dominant_hand_keypoints = np.logical_and(np.ones_like(keypoint3D_left,
                                                          dtype=bool),
                                             dominant_hand)

    hand_side_keypoints = np.where(dominant_hand_keypoints, keypoint3D_left,
                                   keypoint3D_right)

    hand_side = np.where(np.greater(num_pixels_hand_left,
                                    num_pixels_hand_right), 0, 1)

    return hand_side, hand_side_keypoints, dominant_hand_keypoints


def get_canonical_transformations(keypoints3D):
    ROOT_KEYPOINT_ID = 0  #
    ALIGNED_KEYPOINT_ID = 12
    LAST_KEYPOINT_ID = 20
    print(keypoints3D.shape)

    translation_to_root = np.expand_dims(keypoints3D[ROOT_KEYPOINT_ID, :], 0)
    translated_keypoints3D = keypoints3D - translation_to_root

    alignment_keypoint = translated_keypoints3D[:, ALIGNED_KEYPOINT_ID]

    alpha = np.arctan2(alignment_keypoint[0], alignment_keypoint[1])
    rotation_matrix_z = build_rotation_matrix_z(alpha)
    resultant_keypoints3D = np.matmul(translated_keypoints3D.T,
                                      rotation_matrix_z)

    reference_keypoint_z_rotation = \
        resultant_keypoints3D[ALIGNED_KEYPOINT_ID, :]
    beta = -np.arctan2(reference_keypoint_z_rotation[2],
                       reference_keypoint_z_rotation[1])
    rotation_matrix_x = build_rotation_matrix_x(beta + np.pi)
    resultant_keypoints3D = np.matmul(resultant_keypoints3D, rotation_matrix_x)

    reference_keypoint_z_rotation = resultant_keypoints3D[LAST_KEYPOINT_ID, :]
    gamma = np.arctan2(reference_keypoint_z_rotation[2],
                       reference_keypoint_z_rotation[0])
    rotation_matrix_y = build_rotation_matrix_y(gamma)
    transformed_keypoints3D = np.matmul(resultant_keypoints3D,
                                        rotation_matrix_y)

    final_rotation_matrix = np.matmul(np.matmul(rotation_matrix_z,
                                                rotation_matrix_x),
                                      rotation_matrix_y)
    return np.squeeze(transformed_keypoints3D), \
           np.squeeze(final_rotation_matrix)


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


def extract_dominant_keypoints2D(keypoint_2D_visibility, dominant_hand):
    keypoint_visibility_left = keypoint_2D_visibility[:21, :]
    keypoint_visibility_right = keypoint_2D_visibility[-21:, :]
    keypoint_visibility_2D_21 = np.where(dominant_hand[:, :2],
                                         keypoint_visibility_left,
                                         keypoint_visibility_right)
    return keypoint_visibility_2D_21


def crop_image_from_coordinates(image, crop_location, crop_size,
                                scale=0.1, image_size_limit=320):
    crop_size_scaled = crop_size / scale

    y1 = crop_location[0] - crop_size_scaled // 2
    y2 = crop_location[0] + crop_size_scaled // 2
    x1 = crop_location[1] - crop_size_scaled // 2
    x2 = crop_location[1] + crop_size_scaled // 2

    box = [int(x1), int(y1), int(x2), int(y2)]
    box = [max(min(x, image_size_limit), 0) for x in box]

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
    gamma = np.arctan2(vector[0], vector[2])
    rotation_matrix_y = build_rotation_matrix_y(gamma)
    affine_rotation_matrix_y = get_affine_matrix(rotation_matrix_y)

    matrix_after_y_rotation = np.matmul(affine_rotation_matrix_y, vector)
    alpha = np.arctan2(-matrix_after_y_rotation[1],
                       matrix_after_y_rotation[2])

    rotation_matrix_x = build_rotation_matrix_x(alpha)
    affine_rotation_matrix_x = get_affine_matrix(rotation_matrix_x)

    matrix_after_x_rotation = np.matmul(get_translation_matrix(
        -length_from_origin), np.matmul(affine_rotation_matrix_x,
                                        affine_rotation_matrix_y))

    final_transformation_matrix = np.matmul(matrix_after_x_rotation,
                                            transformation_matrix)
    return length_from_origin, alpha, gamma, final_transformation_matrix


def extract_bounding_box(binary_class_mask):
    binary_class_mask = tf.cast(binary_class_mask, tf.int32)
    binary_class_mask = tf.equal(binary_class_mask, 1)
    shape = binary_class_mask.get_shape().as_list()
    if len(shape) == 4:
        binary_class_mask = tf.squeeze(binary_class_mask, [3])

    s = binary_class_mask.get_shape().as_list()
    assert len(s) == 3, "binary_class_mask must be 3D."

    x_range = tf.expand_dims(tf.range(s[1]), 1)
    y_range = tf.expand_dims(tf.range(s[2]), 0)
    X = tf.tile(x_range, [1, s[2]])
    Y = tf.tile(y_range, [s[1], 1])

    bb_list = list()
    center_list = list()
    crop_size_list = list()
    for i in range(s[0]):
        X_masked = tf.cast(tf.boolean_mask(X, binary_class_mask[i, :, :]),
                           tf.float32)
        Y_masked = tf.cast(tf.boolean_mask(Y, binary_class_mask[i, :, :]),
                           tf.float32)

        x_min = tf.reduce_min(X_masked)
        x_max = tf.reduce_max(X_masked)
        y_min = tf.reduce_min(Y_masked)
        y_max = tf.reduce_max(Y_masked)

        start = tf.stack([x_min, y_min])
        end = tf.stack([x_max, y_max])
        bb = tf.stack([start, end], 1)
        bb_list.append(bb)

        center_x = 0.5 * (x_max + x_min)
        center_y = 0.5 * (y_max + y_min)
        center = tf.stack([center_x, center_y], 0)

        center = tf.cond(tf.reduce_all(tf.math.is_finite(center)),
                         lambda: center,
                         lambda: tf.constant([160.0, 160.0]))
        center.set_shape([2])
        center_list.append(center)

        crop_size_x = x_max - x_min
        crop_size_y = y_max - y_min
        crop_size = tf.expand_dims(tf.maximum(crop_size_x, crop_size_y), 0)
        crop_size = tf.cond(tf.reduce_all(tf.math.is_finite(crop_size)),
                            lambda: crop_size,
                            lambda: tf.constant([100.0]))
        crop_size.set_shape([1])
        crop_size_list.append(crop_size)

    bounding_box = tf.stack(bb_list)
    center = tf.stack(center_list)
    crop_size = tf.stack(crop_size_list)

    return center, bounding_box, crop_size


def crop_image_from_xy(image, crop_location, crop_size, scale=1.0):
    s = image.get_shape().as_list()
    assert len(s) == 4, \
        "Image needs to be of shape [batch, width, height, channel]"
    scale = tf.reshape(scale, [-1])
    crop_location = tf.cast(crop_location, tf.float32)
    crop_location = tf.reshape(crop_location, [s[0], 2])
    crop_size = tf.cast(crop_size, tf.float32)

    crop_size_scaled = crop_size / scale

    y1 = crop_location[:, 0] - crop_size_scaled // 2
    y2 = y1 + crop_size_scaled
    x1 = crop_location[:, 1] - crop_size_scaled // 2
    x2 = x1 + crop_size_scaled
    y1 /= s[1]
    y2 /= s[1]
    x1 /= s[2]
    x2 /= s[2]
    boxes = tf.stack([y1, x1, y2, x2], -1)

    crop_size = tf.cast(tf.stack([crop_size, crop_size]), tf.int32)
    box_indices = tf.range(s[0])
    image_cropped = tf.image.crop_and_resize(tf.cast(image, tf.float32), boxes,
                                       box_indices, crop_size, name='crop')
    return image_cropped
