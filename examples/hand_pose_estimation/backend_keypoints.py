import numpy as np
import tensorflow as tf

from backend_SE3 import to_homogeneous_coordinates
from backend_SE3 import build_translation_matrix_SE3
from backend_SE3 import build_rotation_matrix_x, build_rotation_matrix_y
from backend_SE3 import build_rotation_matrix_z, build_affine_matrix

from RHD_v2 import LEFT_ALIGNED_KEYPOINT_ID, LEFT_ROOT_KEYPOINT_ID
from RHD_v2 import LEFT_LAST_KEYPOINT_ID, LEFT_HAND
from RHD_v2 import RIGHT_ALIGNED_KEYPOINT_ID, RIGHT_ROOT_KEYPOINT_ID
from RHD_v2 import RIGHT_LAST_KEYPOINT_ID, RIGHT_HAND
from RHD_v2 import kinematic_chain_dict, kinematic_chain_list


def extract_hand_segment(segmentation_label, hand_arg=1):
    """ Data Pre-processing step: Extract only hand mask from the
    segmentation map provided in RHD dataset.

    # Arguments
        segmentation_label: Numpy array with shape `(320, 320, 1)`.

    # Returns
        Numpy array with shape `(320, 320, 2)`.
    """
    hand_mask = np.greater(segmentation_label, hand_arg)
    background_mask = np.logical_not(hand_mask)
    return np.stack([background_mask, hand_mask], axis=2)


def normalize_keypoints(keypoints3D):
    """ Normalize 3D-keypoints.

    # Arguments
        keypoints: Numpy array with shape `(21, 3)`

    # Returns
        keypoint_scale: Numpy array with shape `(1, )`.
        keypoint_normalized: Numpy array with shape `(21, 3)`.
    """
    keypoint3D_root = keypoints3D[0, :]
    relative_keypoint3D = keypoints3D - keypoint3D_root
    keypoint_scale = np.linalg.norm(
        relative_keypoint3D[LEFT_ALIGNED_KEYPOINT_ID, :] -
        relative_keypoint3D[(LEFT_ALIGNED_KEYPOINT_ID - 1), :])
    keypoint_normalized = relative_keypoint3D / keypoint_scale
    return keypoint_scale, keypoint_normalized


def extract_hand_mask(segmenation_mask, hand_arg=1):
    """ Normalize 3D-keypoints.

    # Arguments
        segmenation_mask: Numpy array with shape `(320, 320)`
        hand_arg: Int value.

    # Returns
        hand_mask: Numpy array with shape `(320, 320)`.
    """
    hand_mask = np.greater(segmenation_mask, hand_arg)
    return hand_mask


def extract_hand_masks(segmentation_mask, right_hand_mask_limit=17):
    """ Extract Hand masks of left and right hand.

    # Arguments
        segmentation_mask: Numpy array of size [320, 320].
        right_hand_mask_limit: Int value.

    # Returns
        mask_left: Numpy array of size (320, 320).
        mask_right: Numpy array of size (320, 320).
    """
    ones_mask = np.ones_like(segmentation_mask)
    hand_mask = extract_hand_mask(segmentation_mask, hand_arg=1)
    right_hand_map = np.less(
        segmentation_mask, ones_mask * (right_hand_mask_limit + 1))
    mask_left = np.logical_and(hand_mask, right_hand_map)
    mask_right = np.greater(segmentation_mask,
                            ones_mask * right_hand_mask_limit)
    return mask_left, mask_right


def extract_hand_side_keypoints(keypoints3D, Is_Left):
    """ Extract Hand masks of dominant hand.

    # Arguments
        keypoints3D: numpy array of shape (21, 3)
        Is_Left: numpy array of shape (1).

    # Returns
        keypoints3D: Numpy array of size (21, 3).
    """
    if Is_Left:
        keypoints3D = keypoints3D[
                      LEFT_ROOT_KEYPOINT_ID:LEFT_LAST_KEYPOINT_ID, :]
    else:
        keypoints3D = keypoints3D[
                      RIGHT_ROOT_KEYPOINT_ID:RIGHT_LAST_KEYPOINT_ID, :]
    return keypoints3D


def get_hand_side_and_keypooints(hand_parts_mask, keypoints3D):
    """Extract Hand masks, Hand side and keypoints of dominant hand.

    # Arguments
        keypoints3D: numpy array of shape (21, 3).
        hand_parts_mask: numpy array of shape (320, 320).

    # Returns
        hand_side: Numpy array of size (2).
        hand_side_keypoints3D: Numpy array of size (21, 3).
        dominant_hand: numpy array of shape (1).
    """
    hand_map_left, hand_map_right = extract_hand_masks(hand_parts_mask)
    num_pixels_hand_left = np.sum(hand_map_left)
    num_pixels_hand_right = np.sum(hand_map_right)
    is_left_dominant = num_pixels_hand_left > num_pixels_hand_right
    if num_pixels_hand_left > num_pixels_hand_right:
        dominant_hand = LEFT_HAND
        keypoints3D = extract_hand_side_keypoints(keypoints3D, True)
    else:
        dominant_hand = RIGHT_HAND
        keypoints3D = extract_hand_side_keypoints(keypoints3D, False)
    hand_side = np.where(is_left_dominant, 0, 1)
    return hand_side, keypoints3D, dominant_hand


def extract_coordinate_limits(keypoints2D, keypoints2D_visibility,
                              image_size):
    """ Extract minimum and maximum coordinates.

    # Arguments
        keypoints2D: Numpy array of shape (21, 2).
        keypoints2D_visibility: Numpy array of shape (21, 1).
        image_size: List of shape (3).

    # Returns
        min_coordinates: Tuple of size (2).
        max_coordinates: Tuple of size (2).
    """
    keypoint_u = keypoints2D[:, 1][keypoints2D_visibility]
    keypoint_v = keypoints2D[:, 0][keypoints2D_visibility]
    keypoints2D_coordinates = np.stack([keypoint_u, keypoint_v], 1)
    min_coordinates = np.maximum(np.amin(keypoints2D_coordinates, 0), 0.0)
    max_coordinates = np.minimum(
        np.amax(keypoints2D_coordinates, 0), image_size[0:2])
    return min_coordinates, max_coordinates


def get_keypoints_camera_coordinates(keypoints2D, crop_center, scale,
                                     crop_size):
    """ Extract keypoints in cropped image frame.

    # Arguments
        keypoints2D: Numpy array of shape (21, 2).
        crop_center: Typle of size (2).
        Scale: Integer.
        image_size: List of size (3).

    # Returns
        keypoint_uv21: Numpy array of shape (21, 2).
    """
    keypoint_u = ((keypoints2D[:, 0] - crop_center[1]) *
                  scale) + (crop_size // 2)
    keypoint_v = ((keypoints2D[:, 1] - crop_center[0]) *
                  scale) + (crop_size // 2)
    keypoint_uv21 = np.stack([keypoint_u, keypoint_v], 1)
    return keypoint_uv21


def get_best_crop_size(max_coordinates, min_coordinates, crop_center):
    """ calculate crop size.

    # Arguments
        max_coordinates: (x_max, y_max) Numpy array of shape (1,2).
        min_coordinates: (x_min, y_min) Numpy array of shape (1,2).
        crop_center: (x_center, y_center) Numpy array of shape (1,2).

    # Returns
        crop_size_best: Int value.
    """
    crop_size_best = 2 * np.maximum(max_coordinates - crop_center,
                                    crop_center - min_coordinates)
    crop_size_best = np.amax(crop_size_best)
    crop_size_best = np.minimum(np.maximum(crop_size_best, 50.0), 500.0)
    if not np.isfinite(crop_size_best):
        crop_size_best = 200.0
    return crop_size_best


def get_crop_scale_and_center(keypoints2D, keypoints2D_visibility, image_size,
                              crop_size):
    """ Extract scale to which image should be cropped.

    # Arguments
        keypoints2D: Numpy array of shape (21, 2).
        keypoints2D_vis: Numpy array of shape (21, 1).
        image_size: List of size (3).
        image_size: List of size (2).

    # Returns
        scale: Integer value.
        crop_center: Tuple of length 3.
    """
    crop_center = keypoints2D[LEFT_ALIGNED_KEYPOINT_ID, ::-1]
    crop_center = np.reshape(crop_center, [2, ])
    min_coordinates, max_coordinates = extract_coordinate_limits(
        keypoints2D, keypoints2D_visibility, image_size)
    crop_size_best = get_best_crop_size(
        max_coordinates, min_coordinates, crop_center)
    scale = crop_size / crop_size_best
    return scale, crop_center


def crop_image_using_mask(keypoints2D, keypoints2D_visibility, image,
                          image_size, crop_size, camera_matrix):
    """ Crop image from mask.

    # Arguments
        keypoints2D: Numpy array of shape (21, 2).
        keypoints2D_vis: Numpy array of shape (21, 1).
        image: Numpy array of shape (320, 320, 3).
        image_size: List of size (2).
        crop_size: List of size (2).
        camera_matrix: Numpy array of shape (3, 3).

    # Returns
        scale: Integer value.
        img_crop: Numpy array of size (256, 256, 3).
        keypoint_uv21: Numpy array of shape (21, 2).
        camera_matrix_cropped: Numpy array of shape (3, 3).
    """
    scale, crop_center = get_crop_scale_and_center(
        keypoints2D, keypoints2D_visibility, image_size, crop_size)
    scale, scale_matrix = get_scale_matrix(scale)
    image_crop = crop_image_from_coordinates(
        image, crop_center, crop_size, scale)
    keypoint_uv21 = get_keypoints_camera_coordinates(
        keypoints2D, crop_center, scale, crop_size)
    scale_translation_matrix = get_scale_translation_matrix(
        crop_center, crop_size, scale)
    scale_matrix_uv = np.matmul(scale_matrix, camera_matrix)
    camera_matrix_cropped = np.matmul(scale_translation_matrix, scale_matrix_uv)
    return scale, np.squeeze(image_crop), keypoint_uv21, camera_matrix_cropped


def flip_right_hand(canonical_keypoints3D, flip_right):
    """ Flip right hend to left hand coordinates.

    # Arguments
        canonical_keypoints3D: Numpy array of shape (21, 3).
        flip_right: boolean value.

    # Returns
        canonical_keypoints3D_left: Numpy array of shape (21, 3).
    """
    shape = canonical_keypoints3D.shape
    expanded = False
    if len(shape) == 2:
        canonical_keypoints3D = np.expand_dims(canonical_keypoints3D, 0)
        flip_right = np.expand_dims(flip_right, 0)
        expanded = True
    canonical_keypoints3D_mirrored = np.stack(
        [canonical_keypoints3D[:, :, 0], canonical_keypoints3D[:, :, 1],
         -canonical_keypoints3D[:, :, 2]], -1)
    canonical_keypoints3D_left = np.where(
        flip_right, canonical_keypoints3D_mirrored, canonical_keypoints3D)
    if expanded:
        canonical_keypoints3D_left = np.squeeze(canonical_keypoints3D_left,
                                                axis=0)
    return canonical_keypoints3D_left


def extract_dominant_hand_visibility(keypoint_visibility, dominant_hand):
    """ Extract Visibility mask for dominant hand.

    # Arguments
        keypoint_visibility: Numpy array of shape (21, 1).
        dominant_hand: List of size (2).

    # Returns
        keypoint_visibility_21: Numpy array of shape (21, 2).
    """
    keypoint_visibility_left = keypoint_visibility[:LEFT_LAST_KEYPOINT_ID + 1]
    keypoint_visibility_right = keypoint_visibility[RIGHT_ROOT_KEYPOINT_ID:
                                                    RIGHT_LAST_KEYPOINT_ID + 1]
    keypoint_visibility_21 = np.where(
        dominant_hand[:, 0], keypoint_visibility_left,
        keypoint_visibility_right)
    return keypoint_visibility_21


def extract_dominant_keypoints2D(keypoint_2D, dominant_hand):
    """ Extract keypoint 2D.

    # Arguments
        keypoint_2D: Numpy array of shape (21, 2).
        dominant_hand: List of size (2) with booleans.

    # Returns
        keypoint_visibility_2D_21: Numpy array of shape (21, 2).
    """
    keypoint_visibility_left = keypoint_2D[:LEFT_LAST_KEYPOINT_ID + 1, :]

    keypoint_visibility_right = keypoint_2D[RIGHT_ROOT_KEYPOINT_ID:
                                            RIGHT_LAST_KEYPOINT_ID + 1, :]
    keypoint_visibility_2D_21 = np.where(
        dominant_hand[:, :2], keypoint_visibility_left,
        keypoint_visibility_right)
    return keypoint_visibility_2D_21


def extract_keypoint2D_limits(uv_coordinates, scoremap_size):
    """ Limit keypoint coordinates to scoremap size ,

    # Arguments
        uv_coordinates: Numpy array of shape (21, 2).
        scoremap_size: List of size (2).

    # Returns
        keypoint_limits: Numpy array of shape (21, 1).
    """
    x_lower_limits = np.less(uv_coordinates[:, 0], scoremap_size[0] - 1)
    x_upper_limits = np.greater(uv_coordinates[:, 0], 0)
    x_limits = np.logical_and(x_lower_limits, x_upper_limits)
    y_lower_limits = np.less(uv_coordinates[:, 1], scoremap_size[1] - 1)
    y_upper_limits = np.greater(uv_coordinates[:, 1], 0)
    y_limits = np.logical_and(y_lower_limits, y_upper_limits)
    keypoint_limits = np.logical_and(x_limits, y_limits)
    return keypoint_limits


def get_keypoints_mask(validity_mask, uv_coordinates, scoremap_size):
    """ Extract Visibility mask for dominant hand.

    # Arguments
        validity_mask: Int value.
        uv_coordinates: Numpy array of shape (21, 2).
        scoremap_size: List of size (2).

    # Returns
        keypoint_limits: Numpy array of shape (21, 1).
    """
    validity_mask = np.squeeze(validity_mask)
    keypoint_validity = np.greater(validity_mask, 0.5)
    keypoint_limits = extract_keypoint2D_limits(uv_coordinates, scoremap_size)
    keypooints_mask = np.logical_and(keypoint_validity, keypoint_limits)
    return keypooints_mask


def get_keypoint_limits(uv_coordinates, scoremap_size):
    """ Extract X and Y limits.

    # Arguments
        uv_coordinates: Numpy array of shape (21, 2).
        scoremap_size: List of size (2).

    # Returns
        X_limits: Numpy array of shape (21, 1).
        Y_limits: Numpy array of shape (21, 1).
    """
    shape = uv_coordinates.shape

    x_range = np.expand_dims(np.arange(scoremap_size[0]), 1)
    x_coordinates = np.tile(x_range, [1, scoremap_size[1]])
    x_coordinates.reshape((scoremap_size[0], scoremap_size[1]))
    x_coordinates = np.expand_dims(x_coordinates, -1)
    x_coordinates = np.tile(x_coordinates, [1, 1, shape[0]])
    x_limits = x_coordinates - uv_coordinates[:, 0].astype('float64')

    y_range = np.expand_dims(np.arange(scoremap_size[1]), 0)
    y_coordinates = np.tile(y_range, [scoremap_size[0], 1])
    y_coordinates.reshape((scoremap_size[0], scoremap_size[1]))
    y_coordinates = np.expand_dims(y_coordinates, -1)
    y_coordinates = np.tile(y_coordinates, [1, 1, shape[0]])
    y_limits = y_coordinates - uv_coordinates[:, 1].astype('float64')

    return x_limits, y_limits


def create_multiple_gaussian_map(uv_coordinates, scoremap_size, sigma,
                                 validity_mask):
    """ Generate Gaussian maps based on keypoints in Image coordinates.

    # Arguments
        uv_coordinates: Numpy array of shape (21, 2).
        scoremap_size: List of size (2).
        sigma: Integer value.
        validity_mask: Integer value.

    # Returns
        scoremap: Numpy array of shape (256, 256).
    """
    assert len(scoremap_size) == 2
    keypoints_mask = get_keypoints_mask(
        validity_mask, uv_coordinates, scoremap_size)
    x_limits, y_limits = get_keypoint_limits(uv_coordinates, scoremap_size)
    squared_distance = np.square(x_limits) + np.square(y_limits)
    scoremap = np.exp(-squared_distance / np.square(sigma)) * keypoints_mask
    return scoremap


def keypoint_xy_coordinates(shape):
    """ Generate X and Y nesh.

    # Arguments
        shape: tuple of size (3).

    # Returns
        X: Numpy array of shape (1, 256).
        Y: Numpy array of shape (256, 1).
    """
    x_range = np.expand_dims(np.arange(shape[1]), 1)
    y_range = np.expand_dims(np.arange(shape[2]), 0)
    x_coordinates = np.tile(x_range, [1, shape[2]])
    y_coordinates = np.tile(y_range, [shape[1], 1])
    return x_coordinates, y_coordinates


def get_bounding_box_list(X_masked, Y_masked):
    """ Get Bounding Box.

    # Arguments
        X_masked: tuple of size (256, 1).
        Y_masked: tuple of size (256, 1).

    # Returns
        bounding_box: List of length (4).
    """
    x_min, x_max = np.min(X_masked), np.max(X_masked)
    y_min, y_max = np.min(Y_masked), np.max(Y_masked)
    start = np.stack([x_min, y_min])
    end = np.stack([x_max, y_max])
    bounding_box = np.stack([start, end], 1)
    return bounding_box


def get_center_list(box_coordinates, image_center=[160, 160]):
    """ Extract Center.

    # Arguments
        box_coordinates: List of length 4.
        center_list: List of length batch_size.

    # Returns
        center_list: List of length batch_size.
    """
    x_min, x_max = box_coordinates[0][0], box_coordinates[0][1]
    y_min, y_max = box_coordinates[1][0], box_coordinates[1][1]
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    center = np.stack([center_x, center_y], 0)
    if not np.all(np.isfinite(center)):
        center = np.array(image_center)
    center.reshape([2])
    return center


def get_crop_list(box_coordinates):
    """ Extract Crop.

    # Arguments
        xy_limit: List of length 4.
        crop_size_list: List of length batch_size.

    # Returns
        crop_size_list: List of length batch_size.
    """
    x_max, x_min = box_coordinates[0][1], box_coordinates[0][0]
    y_max, y_min = box_coordinates[1][1], box_coordinates[1][0]
    crop_size_x = x_max - x_min
    crop_size_y = y_max - y_min
    crop_maximum_value = np.maximum(crop_size_x, crop_size_y)
    crop_size = np.expand_dims(crop_maximum_value, 0)
    crop_size.reshape([1])
    return crop_size


def get_bounding_box_features(X, Y, binary_class_mask, shape):
    """ Extract Crop.

    # Arguments
        X: Numpy array of size (21, 1).
        Y: Numpy array of size (21, 1).
        binary_class_mask: Numpy array of size (320, 320).
        shape: Tuple of lenth (3).

    # Returns
        bounding_box_list: List of length batch_size.
        center_list: List of length batch_size.
        crop_size_list: List of length batch_size.
    """
    bounding_box_list, center_list, crop_size_list = [], [], []
    for binary_class_index in range(shape[0]):
        X_masked = X[binary_class_mask[binary_class_index, :, :]]
        Y_masked = Y[binary_class_mask[binary_class_index, :, :]]
        if len(X_masked) == 0:
            bounding_box_list, center_list, crop_size_list = None, None, None
            return bounding_box_list, center_list, crop_size_list
        bounding_box = get_bounding_box_list(X_masked, Y_masked)
        center = get_center_list(bounding_box)
        crop_size = get_crop_list(bounding_box)
        bounding_box_list.append(bounding_box)
        center_list.append(center)
        crop_size_list.append(crop_size)
    return bounding_box_list, center_list, crop_size_list


def extract_bounding_box(binary_class_mask):
    """ Extract Bounding Box from Segmentation mask.

    # Arguments
        binary_class_mask: Numpy array of size (320, 320).

    # Returns
        bounding_box: Numpy array of shape (batch_size, 4).
        center: Numpy array of shape (batch_size, 2).
        crop_size: Numpy array of shape (batch_size, 1).
    """
    binary_class_mask = np.equal(binary_class_mask, 1)
    shape = binary_class_mask.shape
    if len(shape) == 4:
        binary_class_mask = np.squeeze(binary_class_mask, axis=-1)
        shape = binary_class_mask.shape
    assert len(shape) == 3, "binary_class_mask must be 3D."
    coordinates_x, coordinates_y = keypoint_xy_coordinates(shape)
    bounding_box_list, center_list, crop_size_list = get_bounding_box_features(
        coordinates_x, coordinates_y, binary_class_mask, shape)
    bounding_box = np.stack(bounding_box_list)
    center = np.stack(center_list)
    crop_size = np.stack(crop_size_list)
    return center, bounding_box, crop_size


def get_box_coordinates(location, size, shape):
    """ Extract Bounding Box from center and size of cropped image.

    # rename to get_box_coordinates

    # Arguments
        location: Tuple of length (2).
        size: Tuple of length (2).
        shape: Typle of length (3).

    # Returns
        boxes: Numpy array of shape (batch_size, 4).
    """
    height, width = shape[1], shape[2]
    x_min = location[:, 0] - size // 2
    x_max = x_min + size
    y_min = location[:, 1] - size // 2
    y_max = y_min + size
    x_min = x_min / height
    x_max = x_max / height
    y_min = y_min / width
    y_max = y_max / width
    boxes = np.stack([x_min, y_min, x_max, y_max], -1)
    return boxes


def crop_image(image, crop_box):
    """ Resize image.

    # Arguments
        image: Numpy array.
        crop_box: List of four ints.

    # Returns
        Numpy array.
    """
    if type(image) != np.ndarray:
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        cropped_image = image[
                        crop_box[0]:crop_box[2], crop_box[1]:crop_box[3], :]
    return cropped_image


def crop_image_from_coordinates(image, crop_center, crop_size, scale=1.0):
    """ Crop Image from Center and crop size.

    # Arguments
        Image: Numpy array of shape (320, 320, 3).
        crop_center: Tuple of length (2).
        crop_size: Float.
        Scale: Float.

    # Returns
        Image_cropped: Numpy array of shape (256, 256).
    """
    image_dimensions = image.shape
    scale = np.reshape(scale, [-1])
    crop_location = crop_center.astype(np.float)
    crop_location = np.reshape(crop_location, [image_dimensions[0], 2])
    crop_size = np.float(crop_size)
    crop_size_scaled = crop_size / scale
    boxes = get_box_coordinates(
        crop_location, crop_size_scaled, image_dimensions)
    crop_size = np.stack([crop_size, crop_size])
    crop_size = crop_size.astype(np.float)
    box_indices = np.arange(image_dimensions[0])
    image_cropped = tf.image.crop_and_resize(
        tf.cast(image, tf.float32), boxes, box_indices, crop_size, name='crop')
    return image_cropped.numpy()


def extract_keypoint_index(scoremap):
    """ Extract Scoremap.

    # Arguments
        scoremap: Numpy aray of shape (256, 256).

    # Returns
        max_index_vec: List of Max Indices.
    """
    shape = scoremap.shape
    scoremap = np.reshape(scoremap, [shape[0], -1])
    keypoint_index = np.argmax(scoremap, axis=1)
    return keypoint_index


def extract_keypoints_XY(x_vector, y_vector, maximum_indices, batch_size):
    """ Extract Keypoint X,Y coordinates.

    # Arguments
        x_vector: Numpy array of shape (batch_size, 1).
        y_vector: Numpy array of shape (batch_size, 1).
        maximum_indices: Numpy array of shape (batch_size, 1).
        batch_size: Integer Value.

    # Returns
        keypoints2D: Numpy array of shape (21, 2).
    """
    keypoints2D = list()
    for image_index in range(batch_size):
        index_choice = maximum_indices[image_index]
        x_location = np.reshape(x_vector[index_choice], [1])
        y_location = np.reshape(y_vector[index_choice], [1])
        keypoints2D.append(np.concatenate([x_location, y_location], 0))
    keypoints2D = np.stack(keypoints2D, 0)
    return keypoints2D


def create_2D_grids(shape):
    """ Create 2D Grids.

    # Arguments
        shape: Tuple of length 2.

    # Returns
        x_vec: Numpy array.
        y_vec: Numpy array.
    """
    x_range = np.expand_dims(np.arange(shape[1]), 1)
    y_range = np.expand_dims(np.arange(shape[2]), 0)
    X = np.tile(x_range, [1, shape[2]])
    Y = np.tile(y_range, [shape[1], 1])
    X = np.reshape(X, [-1])
    Y = np.reshape(Y, [-1])
    return X, Y


def find_max_location(scoremap):
    """ Returns the coordinates of the given scoremap with maximum value.

    # Arguments
        scoremap: Numpy array of shape (256, 256).

    # Returns
        keypoints2D: numpy array of shape (21, 2).
    """
    shape = scoremap.shape
    assert len(shape) == 3, "Scoremap must be 3D."
    x_grid, y_grid = create_2D_grids(shape)
    keypoint_index = extract_keypoint_index(scoremap)
    keypoints2D = extract_keypoints_XY(x_grid, y_grid, keypoint_index, shape[0])
    return keypoints2D


def create_score_maps(keypoint_2D, keypoint_visibility, image_size,
                      crop_size, variance, crop_image=True):
    """ Create gaussian maps for keypoint representation.

    # Arguments
        keypoint_2D: Numpy array of shape (21, 2).
        keypoint_vis21: Numpy array of shape (21, 2).
        image_size: Tuple of length (3).
        crop_size: Typle of length (2).
        variance: Float value.
        crop_image: Boolean value.

    # Returns
        scoremap: numpy array of size (21, 256, 256).
    """
    keypoint_uv = np.stack([keypoint_2D[:, 1], keypoint_2D[:, 0]], -1)
    scoremap_size = image_size[0:2]
    if crop_image:
        scoremap_size = (crop_size, crop_size)
    scoremap = create_multiple_gaussian_map(
        keypoint_uv, scoremap_size, variance, keypoint_visibility)
    return scoremap


def extract_2D_keypoints(visibility_mask):
    """ Extract 2D keypoints.

    # Arguments
        visibility_mask: Numpy array of size (21, 3).

    # Returns
        keypoints2D: numpy array of size (21, 2).
        keypoints_visibility_mask: numpy array of size (21, 1).
    """
    keypoints2D = visibility_mask[:, :2]
    keypoints_visibility_mask = visibility_mask[:, 2] == 1
    return keypoints2D, keypoints_visibility_mask


def extract_keypoints(scoremaps):
    """ Performs detection per scoremap for the hands keypoints.

    # Arguments
        scoremaps: Numpy array of size (256, 256, 21).

    # Returns
        keypoint_coords: numpy array of size (21, 2).
    """
    scoremaps = np.squeeze(scoremaps, axis=0)
    height, width, num_keypoints = scoremaps.shape
    keypoint2D = np.zeros((num_keypoints, 2))
    for keypoint_arg in range(num_keypoints):
        keypoint_scoremap = np.argmax(scoremaps[:, :, keypoint_arg])
        coordinates = np.unravel_index(keypoint_scoremap, (height, width))
        keypoint2D[keypoint_arg, 0] = coordinates[1]
        keypoint2D[keypoint_arg, 1] = coordinates[0]
    return keypoint2D


def transform_visibility_mask(visibility_mask):
    """ Data Pre-processing step: Transform Visibility mask to palm coordinates
    from wrist coordinates.

    # Arguments
        visibility_mask: Numpy array with shape `(42, 1)`.

    # Returns
        visibility_mask: Numpy array with shape `(42, 1)`.
    """
    visibility_left_root = visibility_mask[LEFT_ROOT_KEYPOINT_ID]
    visibility_left_aligned = visibility_mask[LEFT_ALIGNED_KEYPOINT_ID]
    visibility_right_root = visibility_mask[RIGHT_ROOT_KEYPOINT_ID]
    visibility_right_aligned = visibility_mask[RIGHT_ALIGNED_KEYPOINT_ID]

    palm_visibility_left = np.logical_or(
        visibility_left_root, visibility_left_aligned)
    palm_visibility_left = np.expand_dims(palm_visibility_left, 0)

    palm_visibility_right = np.logical_or(
        visibility_right_root, visibility_right_aligned)
    palm_visibility_right = np.expand_dims(palm_visibility_right, 0)

    visibility_mask = np.concatenate(
        [palm_visibility_left,
         visibility_mask[LEFT_ROOT_KEYPOINT_ID + 1: LEFT_LAST_KEYPOINT_ID + 1],
         palm_visibility_right,
         visibility_mask[RIGHT_ROOT_KEYPOINT_ID + 1: RIGHT_LAST_KEYPOINT_ID + 1]
         ], 0)

    return visibility_mask


def keypoints_to_palm_coordinates(keypoints):
    """ Data Pre-processing step: Transform keypoints to palm coordinates
        from wrist coordinates.

    # Arguments
        keypoints: Numpy array with shape `(42, 3)` for 3D keypoints.
                   Numpy array with shape `(42, 2)` for 2D keypoints.

    # Returns
        keypoints: Numpy array with shape `(42, 3)` for 3D keypoints.
                   Numpy array with shape `(42, 2)` for 2D keypoints.
    """
    palm_coordinates_left = 0.5 * (keypoints[LEFT_ROOT_KEYPOINT_ID, :] +
                                   keypoints[LEFT_ALIGNED_KEYPOINT_ID, :])
    palm_coordinates_left = np.expand_dims(palm_coordinates_left, 0)

    palm_coordinates_right = 0.5 * (keypoints[RIGHT_ROOT_KEYPOINT_ID, :] +
                                    keypoints[RIGHT_ALIGNED_KEYPOINT_ID, :])
    palm_coordinates_right = np.expand_dims(palm_coordinates_right, 0)

    keypoints = np.concatenate(
        [palm_coordinates_left,
         keypoints[(LEFT_ROOT_KEYPOINT_ID + 1):(LEFT_LAST_KEYPOINT_ID + 1), :],
         palm_coordinates_right,
         keypoints[(RIGHT_ROOT_KEYPOINT_ID + 1):(RIGHT_LAST_KEYPOINT_ID + 1), :]
         ], 0)

    return keypoints


def get_transform_to_bone_frame(keypoints3D, bone_index):
    # bone
    """ Transform the keypoints in camera image frame to index keypoint frame.

    # Arguments
        keypoints3D: numpy array of shape (21, 3).
        bone_index: int value of range [0, 21].

    # Returns
        transformation_parameters: multiple values representing all the
        euclidean parameters to calculate transformation matrix.
    """
    index_keypoint = np.expand_dims(keypoints3D[bone_index, :], 1)
    translated_keypoint3D = to_homogeneous_coordinates(index_keypoint)
    translation_matrix = build_translation_matrix_SE3(
        np.zeros_like(keypoints3D[0, 0]))
    transformation_parameters = get_transformation_parameters(
        translated_keypoint3D, translation_matrix)
    return transformation_parameters


def transform_to_keypoint_coordinates(transformation_matrix, keypoint3D):
    """ Transform to keypoint (root/child) frame.

    # Arguments
        transformation_matrix: numpy array of shape (4, 4).
        keypoint3D: numpy array of shape (3, ).

    # Returns
        keypoint_coordinates: Numpy array of size (3, ).
    """
    keypoint3D = np.expand_dims(keypoint3D, 1)
    keypoint3D = to_homogeneous_coordinates(keypoint3D)
    keypoint_coordinates = np.matmul(transformation_matrix, keypoint3D)
    return keypoint_coordinates


def get_root_transformations(keypoints3D, bone_index):
    """ Transform all keypoints to root keypoint frame.

    # Arguments
        keypoints3D: numpy array of shape (21, 3).
        bone_index: int value of range [0, 21].

    # Returns
        relative_coordinates: numpy array of shape (21, 3, 1).
        transformations: placeholder for transformation (21, 4, 4, 1).
    """
    length_from_origin, rotation_angle_x, rotation_angle_y, rotated_keypoints = \
        get_transform_to_bone_frame(keypoints3D, bone_index)
    relative_coordinate = np.stack(
        [length_from_origin, rotation_angle_x, rotation_angle_y], 0)
    transformation = rotated_keypoints
    return transformation, relative_coordinate


def get_articulation_angles(child_keypoint_coordinates,
                            parent_keypoint_coordinates, transformation_matrix):
    """ Calculate Articulation Angles.

    # Arguments
        local_child_coordinates: Child keypoint coordinates (1, 3).
        local_child_coordinates: Parent keypoint coordinates (1, 3).
        transformation_matrix: Numpy array of shape (4, 4).

    # Returns
        transformation_parameters: parameters for transformation to
        local frame.
    """
    delta_vector = child_keypoint_coordinates - parent_keypoint_coordinates
    delta_vector = to_homogeneous_coordinates(
        np.expand_dims(delta_vector[:, :3], 1))
    transformation_parameters = get_transform_to_bone_frame(
        delta_vector, transformation_matrix)
    return transformation_parameters


def get_child_transformations(keypoints3D, bone_index, parent_index,
                              transformations):
    """ Calculate Child coordinate to Parent coordinate.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).
        bone_index: Index of current bone keypoint, Numpy array of shape (1).
        parent_index: Index of root keypoint, Numpy array of shape (1).
        relative_coordinates: place holder for relative_coordinates.
        transformations: placeholder for transformations.

    # Returns
        relative_coordinates: place holder for relative_coordinates.
        transformations: placeholder for transformations.
    """
    transformation_matrix = transformations[parent_index]
    parent_keypoint_coordinates = transform_to_keypoint_coordinates(
        transformation_matrix, keypoints3D[parent_index, :])
    child_keypoint_coordinates = transform_to_keypoint_coordinates(
        transformation_matrix, keypoints3D[bone_index, :])
    transformation_parameters = get_articulation_angles(
        parent_keypoint_coordinates, child_keypoint_coordinates,
        transformation_matrix)
    relative_coordinate = np.stack(transformation_parameters[:3])
    transformation = transformation_parameters[3]
    return transformation, relative_coordinate


def keypoints_to_root_frame(keypoints3D):
    """ Convert keypoints to root keypoint coordinates.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).

    # Returns
        relative_coordinates: keypoints in root keypoint coordinate frame.
    """
    transformations = [None] * len(kinematic_chain_list)
    relative_coordinates = np.zeros(len(kinematic_chain_list))
    for bone_index in kinematic_chain_list:
        parent_index = kinematic_chain_dict[bone_index]
        if parent_index == 'root':
            transformation, relative_coordinate = get_root_transformations(
                keypoints3D, bone_index)
        else:
            transformation, relative_coordinate = get_child_transformations(
                keypoints3D, bone_index, parent_index, transformations)
        transformations[bone_index] = transformation
        relative_coordinates[bone_index] = relative_coordinate
    return relative_coordinates


def keypoint_to_root_frame(keypoints3D, num_keypoints=21):
    """ Convert keypoints to root keypoint coordinates.
    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).

    # Returns
        key_point_relative_frame: keypoints in root keypoint coordinate frame.
    """
    keypoints3D = keypoints3D.reshape([num_keypoints, 3])
    relative_coordinates = keypoints_to_root_frame(keypoints3D)
    key_point_relative_frame = np.stack(relative_coordinates, 1)
    return key_point_relative_frame


def get_keypoints_z_rotation(keypoints3D, keypoint):
    """ Rotate Keypoints along z-axis.

    # Arguments
        alignment_keypoint: Keypoint to whose frame transformation is to
        be done, Numpy array of shape (1, 3).
        translated_keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).

    # Returns
        reference_keypoint_z_rotation: Reference keypoint after rotation.
        resultant_keypoints3D: keypoints after rotation.
        rotation_matrix_z: Rotation matrix.
    """
    alpha = np.arctan2(keypoint[0], keypoint[1])
    rotation_matrix = build_rotation_matrix_z(alpha)
    keypoints3D = np.matmul(keypoints3D.T, rotation_matrix)
    keypoint = keypoints3D[LEFT_ALIGNED_KEYPOINT_ID, :]
    return keypoint, rotation_matrix, keypoints3D


def get_keypoints_x_rotation(keypoints3D, keypoint):
    """ Rotate Keypoints along x-axis.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).
        keypoint: Numpy array of shape (1, 3).

    # Returns
        keypoint: Resultant reference keypoint after rotation, Numpy array of
        shape (1, 3).
        resultant_keypoints3D: keypoints after rotation.
        rotation_matrix_x: Rotation matrix along x-axis.
    """
    beta = -np.arctan2(keypoint[2], keypoint[1])
    rotation_matrix = build_rotation_matrix_x(beta + np.pi)
    keypoints3D = np.matmul(keypoints3D, rotation_matrix)
    keypoint = keypoints3D[LEFT_LAST_KEYPOINT_ID, :]
    return keypoint, rotation_matrix, keypoints3D


def get_keypoints_y_rotation(keypoints3D, keypoint):
    """ Rotate Keypoints along y-axis.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).
        reference_keypoint: keypoint, Numpy array of shape (1, 3).

    # Returns
        resultant_keypoint: Resultant reference keypoint after rotation.
        resultant_keypoints3D: keypoints after rotation along Y-axis.
        rotation_matrix_y: Rotation matrix along x-axis.
    """
    gamma = np.arctan2(keypoint[2], keypoint[0])
    rotation_matrix = build_rotation_matrix_y(gamma)
    keypoints3D = np.matmul(keypoints3D, rotation_matrix)
    keypoint = keypoints3D[LEFT_LAST_KEYPOINT_ID, :]
    return keypoint, rotation_matrix, keypoints3D


def canonical_transformations_on_keypoints(keypoints3D):  # rename properly
    """ Transform Keypoints to canonical coordinates.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, 21, 3).

    # Returns
        transformed_keypoints3D: Resultant keypoint after transformation.
        final_rotation_matrix: Final transformation matrix.
    """
    reference_keypoint = np.expand_dims(
        keypoints3D[:, LEFT_ROOT_KEYPOINT_ID], 1)
    keypoints3D = keypoints3D - reference_keypoint
    keypoint = keypoints3D[:, LEFT_ALIGNED_KEYPOINT_ID]
    final_rotation_matrix = np.ones((3, 3))
    apply_rotations = [get_keypoints_z_rotation, get_keypoints_x_rotation,
                       get_keypoints_y_rotation]
    for function in apply_rotations:
        keypoint, rotation_matrix, keypoints3D = function(
            keypoints3D, keypoint)
        final_rotation_matrix = np.matmul(
            final_rotation_matrix, rotation_matrix)
    return np.squeeze(keypoints3D), np.squeeze(final_rotation_matrix)


def get_scale_matrix(scale):
    """ calculate scale matrix.

    # Arguments
        scale: Int value.

    # Returns
        scale_original: Int value
        scale_matrix: Numpy array of shape (3, 3)
    """
    scale_original = np.minimum(np.maximum(scale, 1.0), 10.0)
    scale_matrix = np.eye(3)
    scale = np.reshape(scale_original, [1, ])
    scale_matrix[0][0] = scale
    scale_matrix[1][1] = scale
    return scale_original, scale_matrix


def get_scale_translation_matrix(crop_center, crop_size, scale):
    """ calculate scale translation matrix.

    # Arguments
        crop_center: Numpy array of shape (2).
        crop_size: Int value.
        scale: Int value.

    # Returns
        translation_matrix: Numpy array of shape (3, 3).
    """
    translation_matrix = np.eye(3)
    translated_center_x = crop_center[0] * scale - crop_size // 2
    translated_center_x = np.reshape(translated_center_x, [1, ])

    translated_center_y = crop_center[1] * scale - crop_size // 2
    translated_center_y = np.reshape(translated_center_y, [1, ])

    translation_matrix[0][2] = -translated_center_x
    translation_matrix[1][2] = -translated_center_y

    return translation_matrix


def get_y_axis_rotated_keypoints(keypoint3D):
    """ Generate keypoints in Image coordinates.

    # Arguments
        keypoint3D: Numpy array of shape (21, 3).

    # Returns
        keypoint3D: Numpy array of shape (21, 3).
        affine_rotation_matrix_y: Numpy array of shape (3, 3).
        gamma: Numpy array of shape (1, ).
    """
    gamma = np.arctan2(keypoint3D[0], keypoint3D[2])
    rotation_matrix_y = build_rotation_matrix_y(gamma)
    affine_rotation_matrix_y = build_affine_matrix(rotation_matrix_y)
    keypoint3D = np.matmul(affine_rotation_matrix_y, keypoint3D)
    return keypoint3D, affine_rotation_matrix_y, gamma


def get_x_axis_rotated_keypoints(keypoint3D, length_from_origin,
                                 rotation_matrix):
    """ Generate Gaussian maps based on keypoints in Image coordinates.

    # Arguments
        keypoint3D: Numpy array of shape (21, 3).
        length_from_origin: Numpy array of shape (1, ).
        rotation_matrix: Numpy array of shape (3, 3).

    # Returns
        keypoint3D: Numpy array of shape (21, 3).
        affine_rotation_matrix_y: Numpy array of shape (3, 3).
        gamma: Numpy array of shape (1, ).
    """
    alpha = np.arctan2(-keypoint3D[1], keypoint3D[2])
    rotation_matrix_x = build_rotation_matrix_x(alpha)
    affine_rotation_matrix_x = build_affine_matrix(rotation_matrix_x)
    translation_matrix_to_origin = build_translation_matrix_SE3(
        -length_from_origin)
    rotation_matrix_xy = np.matmul(affine_rotation_matrix_x, rotation_matrix)
    keypoint3D = np.matmul(translation_matrix_to_origin, rotation_matrix_xy)
    return keypoint3D, alpha


def get_transformation_parameters(keypoint3D, transformation_matrix):
    """ Calculate transformation parameters.

    # Arguments
        keypoint3D: Numpy array of shape (21, 3).
        transformation_matrix: Numpy array of shape (4, 4).

    # Returns
        length_from_origin: float value.
        alpha: float value. Rotation angle along X-axis.
        gamma: float value. Rotation angle along X-axis.
        final_transformation_matrix: Numpy array of shape (4, 4).
    """
    length_from_origin = np.linalg.norm(keypoint3D)

    keypoint3D_rotated_y, affine_matrix, \
    rotation_angle_y = get_y_axis_rotated_keypoints(keypoint3D)

    keypoint3D_rotated_x, rotation_angle_x = get_x_axis_rotated_keypoints(
        keypoint3D_rotated_y, length_from_origin, affine_matrix)

    rotated_keypoints = np.matmul(
        keypoint3D_rotated_x, transformation_matrix)

    return length_from_origin, rotation_angle_x, rotation_angle_y, \
           rotated_keypoints


def transform_cropped_keypoints(cropped_keypoints, centers, scale, crop_size):
    """ Transforms the cropped coordinates to the original image space.

    # Arguments
        cropped_coords: Tensor (batch x num_keypoints x 3): Estimated hand
            coordinates in the cropped space.
        centers: Tensor (batch x 1): Repeated coordinates of the
            center of the hand in global image space.
        scale: Tensor (batch x 1): Scaling factor between the original image
            and the cropped image.
        crop_size: int: Size of the crop.

    # Returns
        keypoints: Tensor (batch x num_keypoints x 3): Transformed coordinates.
    """
    keypoints = np.copy(cropped_keypoints)
    keypoints = keypoints - (crop_size // 2)
    keypoints = keypoints / scale
    keypoints = keypoints + centers
    return keypoints
