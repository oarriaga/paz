import numpy as np

from backend_SE3 import to_homogeneous_coordinates
from backend_SE3 import build_translation_matrix_SE3
from backend_SE3 import build_rotation_matrix_x, build_rotation_matrix_y
from backend_SE3 import build_rotation_matrix_z, build_affine_matrix

from RHDv2 import LEFT_MIDDLE_METACARPAL, LEFT_WRIST
from RHDv2 import LEFT_PINKY_TIP, LEFT_HAND
from RHDv2 import RIGHT_MIDDLE_METACARPAL, RIGHT_WRIST
from RHDv2 import RIGHT_PINKY_TIP, RIGHT_HAND
from RHDv2 import KINEMATIC_CHAIN_DICT, KINEMATIC_CHAIN_LIST

from paz.backend.image.opencv_image import resize_image, show_image


def extract_hand_segment(segmentation_label, hand_arg=1):
    """ Data Pre-processing step: Extract only hand mask from the
    segmentation map provided in RHD dataset.

    # Arguments
        segmentation_label: Numpy array.

    # Returns
        Numpy array.
    """
    hand_mask = np.greater(segmentation_label, hand_arg)
    background_mask = np.logical_not(hand_mask)
    return np.stack([background_mask, hand_mask], axis=2)


def normalize_keypoints(keypoints3D):
    """ Normalize 3D-keypoints.

    # Arguments
        keypoints: Numpy array with shape `(num_keypoints, 3)`

    # Returns
        keypoint_scale: Numpy array with shape `(1, )`.
        keypoint_normalized: Numpy array with shape `(num_keypoints, 3)`.
    """
    keypoint3D_root = keypoints3D[0, :]
    relative_keypoint3D = keypoints3D - keypoint3D_root
    metacarpal_bone_length = np.linalg.norm(
        relative_keypoint3D[LEFT_MIDDLE_METACARPAL, :] -
        relative_keypoint3D[(LEFT_MIDDLE_METACARPAL - 1), :])
    keypoint_normalized = relative_keypoint3D / metacarpal_bone_length
    return metacarpal_bone_length, keypoint_normalized


def extract_hand_mask(segmenation_mask, hand_arg=1):
    """ Normalize 3D-keypoints.

    # Arguments
        segmenation_mask: Numpy array
        hand_arg: Int value.

    # Returns
        hand_mask: Numpy array.
    """
    hand_mask = np.greater(segmenation_mask, hand_arg)
    return hand_mask


def extract_hand_masks(segmentation_mask, right_hand_mask_limit=18):
    """ Extract Hand masks of left and right hand.
    ones_mask * right_hand_mask_limit convert to a variable

    # Arguments
        segmentation_mask: Numpy array.
        right_hand_mask_limit: Int value.

    # Returns
        mask_left: Numpy array.
        mask_right: Numpy array.
    """
    ones_mask = np.ones_like(segmentation_mask)
    hand_mask = extract_hand_mask(segmentation_mask, hand_arg=1)
    right_hand_mask = ones_mask * right_hand_mask_limit
    right_hand_map = np.less(segmentation_mask, right_hand_mask)
    mask_left = np.logical_and(hand_mask, right_hand_map)
    mask_right = np.greater(segmentation_mask, right_hand_mask)
    return mask_left, mask_right


def extract_hand_side_keypoints(keypoints3D, dominant_hand):
    """ Extract keypoints related to Left or Right hand.

    # Arguments
        keypoints3D: numpy array of shape (num_keypoints, 3)
        Is_Left: numpy array of shape (1).

    # Returns
        keypoints3D: Numpy array of size (num_keypoints, 3).
    """
    if dominant_hand == LEFT_HAND:
        keypoints3D = keypoints3D[LEFT_WRIST:LEFT_PINKY_TIP, :]
    else:
        keypoints3D = keypoints3D[RIGHT_WRIST:RIGHT_PINKY_TIP, :]
    return keypoints3D


def get_hand_side_and_keypooints(hand_parts_mask, keypoints3D):
    """Extract hand masks, hand side and keypoints of dominant hand.

    # Arguments
        keypoints3D: numpy array of shape (num_keypoints, 3).
        hand_parts_mask: numpy array of shape (image_size, image_size).

    # Returns
        hand_side: Numpy array of size (2).
        hand_side_keypoints3D: Numpy array of size (num_keypoints, 3).
        dominant_hand: numpy array of shape (1).
    """
    hand_map_left, hand_map_right = extract_hand_masks(hand_parts_mask)
    num_pixels_hand_left = np.sum(hand_map_left)
    num_pixels_hand_right = np.sum(hand_map_right)
    is_left_dominant = num_pixels_hand_left > num_pixels_hand_right
    dominant_hand = LEFT_HAND if is_left_dominant else RIGHT_HAND
    keypoints3D = extract_hand_side_keypoints(keypoints3D, dominant_hand)
    hand_side = np.where(is_left_dominant, 0, 1)
    return hand_side, keypoints3D, dominant_hand


def extract_coordinate_limits(keypoints2D, keypoints2D_visibility,
                              image_size):
    """ Extract minimum and maximum coordinates.
    # Try to convert to a function , check numpy.permute , rollaxis, flip
    # Arguments
        keypoints2D: Numpy array of shape (num_keypoints, 2).
        keypoints2D_visibility: Numpy array of shape (num_keypoints, 2).
        image_size: List of shape (3).

    # Returns
        min_coordinates: Tuple of size (2).
        max_coordinates: Tuple of size (2).
    """
    visible_keypoints = keypoints2D[keypoints2D_visibility]
    keypoint_u = visible_keypoints[:, 1]
    keypoint_v = visible_keypoints[:, 0]
    keypoints2D_coordinates = np.stack([keypoint_u, keypoint_v], 1)
    max_keypoint2D = np.maximum(keypoints2D_coordinates, 0)
    min_keypoint2D = np.minimum(keypoints2D_coordinates, 0)
    min_coordinates = np.maximum(min_keypoint2D, 0.0)
    max_coordinates = np.minimum(max_keypoint2D, image_size[0:2])
    return min_coordinates, max_coordinates


def tranform_keypoints_to_camera_coordinates(keypoints2D, crop_center, scale,
                                             crop_size):
    """ Extract keypoints in cropped image frame.

    # Arguments
        keypoints2D: Numpy array of shape (num_keypoints, 1).
        crop_center: Typle of size (2).
        Scale: Integer.
        image_size: List of size (3).

    # Returns
        keypoint_uv21: Numpy array of shape (num_keypoints, 1).
    """
    crop_size_halved = crop_size // 2
    u_residual = keypoints2D[:, 0] - crop_center[1]
    v_residual = keypoints2D[:, 1] - crop_center[0]
    keypoint_u = (u_residual * scale) + crop_size_halved
    keypoint_v = (v_residual * scale) + crop_size_halved
    keypoint_uv = np.stack([keypoint_u, keypoint_v], 1)
    return keypoint_uv


def get_best_crop_size(max_coordinates, min_coordinates, crop_center,
                       min_crop_size=50.0, max_crop_size=500.0):
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
    crop_size_best = np.maximum(crop_size_best)
    crop_size_best = np.minimum(np.maximum(crop_size_best, min_crop_size),
                                max_crop_size)
    return crop_size_best


def get_crop_scale_and_center(keypoints2D, keypoints2D_visibility, image_size,
                              crop_size):
    """ Extract scale to which image should be cropped.

    # Arguments
        keypoints2D: Numpy array of shape (num_keypoints, 1).
        keypoints2D_visibility: Numpy array of shape (num_keypoints, 1).
        image_size: List of size (3).
        crop_size: List of size (2).

    # Returns
        scale: Integer value.
        crop_center: Tuple of length 3.
    """
    crop_center = keypoints2D[LEFT_MIDDLE_METACARPAL, ::-1]
    min_coordinates, max_coordinates = extract_coordinate_limits(
        keypoints2D, keypoints2D_visibility, image_size)
    crop_size_best = get_best_crop_size(max_coordinates, min_coordinates,
                                        crop_center)
    scale = crop_size / crop_size_best
    return scale, crop_center


def crop_image_from_mask(keypoints2D, keypoints2D_visibility, image,
                         image_size, crop_size, camera_matrix):
    """ Crop image from mask.

    # Arguments
        keypoints2D: Numpy array of shape (num_keypoints, 1).
        keypoints2D_vis: Numpy array of shape (num_keypoints, 1).
        image: Numpy array of shape (image_size, image_size, 3).
        image_size: List of size (2).
        crop_size: List of size (2).
        camera_matrix: Numpy array of shape (3, 3).

    # Returns
        scale: Integer value.
        img_crop: Numpy array of size (crop_size, crop-size, 3).
        keypoint_uv21: Numpy array of shape (num_keypoints, 1).
        camera_matrix_cropped: Numpy array of shape (3, 3).
    """
    scale, crop_center = get_crop_scale_and_center(
        keypoints2D, keypoints2D_visibility, image_size, crop_size)
    scale, scale_matrix = get_scale_matrix(scale)
    cropped_image = crop_image_from_coordinates(
        image, crop_center, crop_size, scale)
    keypoint_uv21 = tranform_keypoints_to_camera_coordinates(
        keypoints2D, crop_center, scale, crop_size)
    scale_translation_matrix = get_scale_translation_matrix(
        crop_center, crop_size, scale)
    scale_matrix_uv = np.matmul(scale_matrix, camera_matrix)
    camera_matrix_cropped = np.matmul(scale_translation_matrix, scale_matrix_uv)
    return scale, np.squeeze(
        cropped_image), keypoint_uv21, camera_matrix_cropped


def flip_right_to_left_hand(keypoints3D, flip_right):
    """ Flip right hend coordinates to left hand coordinates.
    # Arguments
        canonical_keypoints3D: Numpy array of shape (num_keypoints, 3).
        flip_right: boolean value.

    # Returns
        canonical_keypoints3D_left: Numpy array of shape (num_keypoints, 3).
    """
    keypoints3D_mirrored = np.stack([keypoints3D[:, 0], keypoints3D[:, 1],
                                     -keypoints3D[:, 2]], -1)
    keypoints3D_left = np.where(flip_right, keypoints3D_mirrored, keypoints3D)
    return keypoints3D_left


def extract_dominant_hand_visibility(keypoint_visibility, dominant_hand):
    """ Extract Visibility mask for dominant hand.
    # Look Later with Octavio
    # Arguments
        keypoint_visibility: Numpy array of shape (num_keypoints, 1).
        dominant_hand: List of size (2).

    # Returns
        keypoint_visibility_21: Numpy array of shape (num_keypoints, 1).
    """
    keypoint_visibility_left = keypoint_visibility[:LEFT_PINKY_TIP]
    keypoint_visibility_right = keypoint_visibility[RIGHT_WRIST:RIGHT_PINKY_TIP]
    keypoint_visibility_21 = np.where(dominant_hand[:, 0],
                                      keypoint_visibility_left,
                                      keypoint_visibility_right)
    return keypoint_visibility_21


def extract_dominant_keypoints2D(keypoint_2D, dominant_hand):
    """ Extract keypoint 2D.
    # Look Later with Octavio
    # Arguments
        keypoint_2D: Numpy array of shape (num_keypoints, 1).
        dominant_hand: List of size (2) with booleans.

    # Returns
        keypoint_visibility_2D_21: Numpy array of shape (num_keypoints, 1).
    """
    keypoint_visibility_left = keypoint_2D[:LEFT_PINKY_TIP, :]
    keypoint_visibility_right = keypoint_2D[RIGHT_WRIST:RIGHT_PINKY_TIP, :]
    keypoint_visibility_2D_21 = np.where(
        dominant_hand[:, :2], keypoint_visibility_left,
        keypoint_visibility_right)
    return keypoint_visibility_2D_21


def extract_keypoint2D_limits(uv_coordinates, scoremap_size):
    """ Limit keypoint coordinates to scoremap size ,
    # Arguments
        uv_coordinates: Numpy array of shape (num_keypoints, 1).
        scoremap_size: List of size (2).

    # Returns
        keypoint_limits: Numpy array of shape (num_keypoints, 1).
    """
    scoremap_height, scoremap_width = scoremap_size
    x_lower_limits = np.less(uv_coordinates[:, 0], scoremap_height - 1)
    x_upper_limits = np.greater(uv_coordinates[:, 0], 0)
    x_limits = np.logical_and(x_lower_limits, x_upper_limits)

    y_lower_limits = np.less(uv_coordinates[:, 1], scoremap_width - 1)
    y_upper_limits = np.greater(uv_coordinates[:, 1], 0)
    y_limits = np.logical_and(y_lower_limits, y_upper_limits)

    keypoint_limits_mask = np.logical_and(x_limits, y_limits)
    return keypoint_limits_mask


def get_keypoints_mask(validity_mask, uv_coordinates, scoremap_size,
                       validity_score=0.5):
    """ Extract Visibility mask for dominant hand.
    # Add in dataset README the difference between seg and vis
    # Arguments
        validity_mask: Int value.
        uv_coordinates: Numpy array of shape (num_keypoints, 1).
        scoremap_size: List of size (2).

    # Returns
        keypoint_limits: Numpy array of shape (num_keypoints, 1).
    """
    validity_mask = np.squeeze(validity_mask)
    keypoint_validity = np.greater(validity_mask, validity_score)
    keypoint_limits = extract_keypoint2D_limits(uv_coordinates, scoremap_size)
    keypooints_mask = np.logical_and(keypoint_validity, keypoint_limits)
    return keypooints_mask


def get_keypoint_limits(uv_coordinates, scoremap_size):
    """ Extract X and Y limits.
    # Arguments
        uv_coordinates: Numpy array of shape (num_keypoints, 2).
        scoremap_size: List of size (2).

    # Returns
        X_limits: Numpy array of shape (num_keypoints, 1).
        Y_limits: Numpy array of shape (num_keypoints, 1).
    """
    shape = uv_coordinates.shape
    scoremap_height, scoremap_width = scoremap_size

    x_range = np.expand_dims(np.arange(scoremap_height), 1)
    x_coordinates = np.tile(x_range, [1, scoremap_width])
    x_coordinates.reshape((scoremap_height, scoremap_width))
    x_coordinates = np.expand_dims(x_coordinates, -1)
    x_coordinates = np.tile(x_coordinates, [1, 1, shape[0]])
    x_limits = x_coordinates - uv_coordinates[:, 0].astype('float64')

    y_range = np.expand_dims(np.arange(scoremap_width), 0)
    y_coordinates = np.tile(y_range, [scoremap_height, 1])
    y_coordinates.reshape((scoremap_height, scoremap_width))
    y_coordinates = np.expand_dims(y_coordinates, -1)
    y_coordinates = np.tile(y_coordinates, [1, 1, shape[0]])
    y_limits = y_coordinates - uv_coordinates[:, 1].astype('float64')

    return x_limits, y_limits


def create_gaussian_map(uv_coordinates, scoremap_size, sigma, validity_mask):
    """ Generate Gaussian maps based on keypoints in Image coordinates.
    # Arguments
        uv_coordinates: Numpy array of shape (num_keypoints, 1).
        scoremap_size: List of size (2).
        sigma: Integer value.
        validity_mask: Integer value.

    # Returns
        scoremap: Numpy array of shape (crop_size, crop-size).
    """
    keypoints_mask = get_keypoints_mask(validity_mask, uv_coordinates,
                                        scoremap_size)
    x_limits, y_limits = get_keypoint_limits(uv_coordinates, scoremap_size)
    squared_distance = np.square(x_limits) + np.square(y_limits)
    scoremap = np.exp(-squared_distance / np.square(sigma)) * keypoints_mask
    return scoremap


def extract_keypoints_uv_coordinates(shape):
    """ Generate X and Y mesh.
    # Rename to best name
    # Arguments
        shape: tuple of size (3).

    # Returns
        X: Numpy array of shape (1, crop_size).
        Y: Numpy array of shape (crop_size, 1).
    """
    crop_size_height, crop_size_width = shape[0], shape[1]
    x_range = np.expand_dims(np.arange(crop_size_height), 1)
    y_range = np.expand_dims(np.arange(crop_size_width), 0)
    x_coordinates = np.tile(x_range, [1, crop_size_width])
    y_coordinates = np.tile(y_range, [crop_size_height, 1])
    return x_coordinates, y_coordinates


def get_bounding_box(X_masked, Y_masked):
    """ Get Bounding Box.

    # Arguments
        X_masked: tuple of size (crop_size, 1).
        Y_masked: tuple of size (crop_size, 1).

    # Returns
        bounding_box: List of length (4).
    """
    x_min, x_max = np.min(X_masked), np.max(X_masked)
    y_min, y_max = np.min(Y_masked), np.max(Y_masked)
    bounding_box = np.array([x_min, y_min, x_max, y_max])
    return bounding_box


def get_crop_center(box_coordinates):
    """ Extract Center.
    # Arguments
        box_coordinates: List of length 4.
        center_list: List of length batch_size.

    # Returns
        center_list: List of length batch_size.
    """
    x_min, x_max = box_coordinates[0], box_coordinates[2]
    y_min, y_max = box_coordinates[1], box_coordinates[3]
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    center = np.stack([center_x, center_y], 0)
    return center


def get_crop_size(box_coordinates):
    """ Extract Crop.

    # Arguments
        xy_limit: List of length 4.
        crop_size_list: List of length batch_size.

    # Returns
        crop_size_list: List of length batch_size.
    """
    x_max, x_min = box_coordinates[2], box_coordinates[0]
    y_max, y_min = box_coordinates[3], box_coordinates[1]
    crop_size_x = x_max - x_min
    crop_size_y = y_max - y_min
    crop_maximum_value = np.maximum(crop_size_x, crop_size_y)
    crop_size = np.expand_dims(crop_maximum_value, 0)
    return crop_size


# RESTART_LINE
def get_bounding_box_features(X, Y, binary_class_mask):
    """ Extract Crop.

    # Arguments
        X: Numpy array of size (num_keypoints, 1).
        Y: Numpy array of size (num_keypoints, 1).
        binary_class_mask: Numpy array of size (image_size, image_size).
        shape: Tuple of lenth (3).

    # Returns
        bounding_box_list: List of length batch_size.
        center_list: List of length batch_size.
        crop_size_list: List of length batch_size.
    """
    X_masked = X[binary_class_mask]
    Y_masked = Y[binary_class_mask]
    bounding_box = get_bounding_box(X_masked, Y_masked)
    center = get_crop_center(bounding_box)
    crop_size = get_crop_size(bounding_box)
    bounding_box = [bounding_box[1],bounding_box[0],bounding_box[3],
                    bounding_box[2]]
    return bounding_box, center, crop_size


def extract_bounding_box(binary_class_mask):
    """ Extract Bounding Box from Segmentation mask.

    # Arguments
        binary_class_mask: Numpy array of size (image_size, image_size).

    # Returns
        bounding_box: Numpy array of shape (batch_size, 4).
        center: Numpy array of shape (batch_size, 2).
        crop_size: Numpy array of shape (batch_size, 1).
    """
    binary_class_mask = binary_class_mask.astype('int')
    binary_class_mask = np.equal(binary_class_mask, 1)
    binary_class_mask = np.squeeze(binary_class_mask, axis=-1)
    shape = binary_class_mask.shape
    coordinates_x, coordinates_y = extract_keypoints_uv_coordinates(shape)
    bounding_box, center, crop_size = get_bounding_box_features(
        coordinates_x, coordinates_y, binary_class_mask)
    return center, bounding_box, crop_size


def get_box_coordinates(center, size, shape):
    """ Extract Bounding Box from center and size of cropped image.

    # Arguments
        location: Tuple of length (2).
        size: Tuple of length (2).
        shape: Typle of length (3).

    # Returns
        boxes: Numpy array of shape (batch_size, 4).
    """
    height, width = shape[0], shape[1]
    x_min = center[0] - size // 2
    y_min = center[1] - size // 2
    x_max, y_max = x_min + size, y_min + size
    x_min, x_max = x_min / height, x_max / height
    y_min, y_max = y_min / width, y_max / width
    boxes = [x_min, y_min, x_max, y_max]
    return boxes


def crop_image_from_coordinates(image, crop_center, crop_size, scale=1.0):
    """ Crop Image from Center and crop size.

    # Arguments
        Image: Numpy array of shape (image_size, image_size, 3).
        crop_center: Tuple of length (2).
        crop_size: Float.
        Scale: Float.

    # Returns
        Image_cropped: Numpy array of shape (crop_size, crop-size).
    """
    image = np.squeeze(image, 0)
    height, width, channels = image.shape
    scale = np.reshape(scale, [-1])
    crop_location = crop_center.astype(np.float)
    crop_size_scaled = crop_size / scale
    boxes = get_box_coordinates(crop_location, crop_size_scaled,
                                image.shape)
    x_min, y_min, x_max, y_max = boxes
    box = [int(x_min * width),
           int(y_min * height),
           int(x_max * width),
           int(y_max * height)]
    image_cropped = crop_image(image, box)
    image_cropped = resize_image(image_cropped, (crop_size, crop_size))
    return image_cropped


def crop_image(image, crop_box):
    """Crop image.

    # Arguments
        image: Numpy array.
        crop_box: List of four ints.

    # Returns
        Numpy array.
    """
    cropped_image = image[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3], :]
    return cropped_image


def extract_keypoint_index(scoremap):
    """ Extract Scoremap.

    # Arguments
        scoremap: Numpy aray of shape (crop_size, crop-size).

    # Returns
        max_index_vec: List of Max Indices.
    """
    keypoint_index = np.argmax(scoremap)
    return keypoint_index


def extract_keypoints_XY(x_vector, y_vector, maximum_indices):
    """ Extract Keypoint X,Y coordinates.
    # Arguments
        x_vector: Numpy array of shape (batch_size, 1).
        y_vector: Numpy array of shape (batch_size, 1).
        maximum_indices: Numpy array of shape (batch_size, 1).
        batch_size: Integer Value.

    # Returns
        keypoints2D: Numpy array of shape (num_keypoints, 1).
    """
    keypoints2D = list()
    x_location = np.reshape(x_vector[maximum_indices], [1])
    y_location = np.reshape(y_vector[maximum_indices], [1])
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
    height, width = shape
    x_range = np.expand_dims(np.arange(height), 1)
    y_range = np.expand_dims(np.arange(width), 0)
    X = np.tile(x_range, [1, width])
    Y = np.tile(y_range, [height, 1])
    X = np.reshape(X, [-1])
    Y = np.reshape(Y, [-1])
    return X, Y


def find_max_location(scoremap):
    """ Returns the coordinates of the given scoremap with maximum value.

    # Arguments
        scoremap: Numpy array of shape (crop_size, crop-size).

    # Returns
        keypoints2D: numpy array of shape (num_keypoints, 1).
    """
    shape = scoremap.shape
    x_grid, y_grid = create_2D_grids(shape)
    keypoint_index = extract_keypoint_index(scoremap)
    keypoints2D = extract_keypoints_XY(x_grid, y_grid, keypoint_index)
    return keypoints2D


def create_score_maps(keypoint_2D, keypoint_visibility, image_size,
                      crop_size, variance, crop_image=True):
    """ Create gaussian maps for keypoint representation.
    # Arguments
        keypoint_2D: Numpy array of shape (num_keypoints, 2).
        keypoint_visibility: Numpy array of shape (num_keypoints, 1).
        image_size: Tuple of length (3).
        crop_size: Typle of length (2).
        variance: Float value.
        crop_image: Boolean value.

    # Returns
        scoremap: numpy array of size (num_keypoints, crop_size, crop-size).
    """
    keypoint_uv = np.stack([keypoint_2D[:, 1], keypoint_2D[:, 0]], -1)
    scoremap_size = image_size[0:2]
    if crop_image:
        scoremap_size = (crop_size, crop_size)
    scoremap = create_gaussian_map(keypoint_uv, scoremap_size, variance,
                                   keypoint_visibility)  # Check if visibility
    # can be removed
    return scoremap


def extract_2D_keypoints(visibility_mask):
    """ Extract 2D keypoints.

    # Arguments
        visibility_mask: Numpy array of size (num_keypoints, 3).

    # Returns
        keypoints2D: numpy array of size (num_keypoints, 1).
        keypoints_visibility_mask: numpy array of size (num_keypoints, 1).
    """
    keypoints2D = visibility_mask[:, :2]
    keypoints_visibility_mask = visibility_mask[:, 2] == 1
    return keypoints2D, keypoints_visibility_mask


def extract_keypoints(scoremaps):
    """ Performs detection per scoremap for the hands keypoints.

    # Arguments
        scoremaps: Numpy array of size (crop_size, crop-size, num_keypoints).

    # Returns
        keypoint_coords: numpy array of size (num_keypoints, 1).
    """
    height, width, num_keypoints = scoremaps.shape
    keypoint2D = np.zeros((num_keypoints, 2))
    for keypoint_arg in range(num_keypoints):
        keypoint_scoremap = np.argmax(scoremaps[:, :, keypoint_arg])
        coordinates = np.unravel_index(keypoint_scoremap, (height, width))
        v, u = coordinates
        keypoint2D[keypoint_arg, 0] = u
        keypoint2D[keypoint_arg, 1] = v
    return keypoint2D


def transform_visibility_mask(visibility_mask):
    """ Data Pre-processing step: Transform Visibility mask to palm coordinates
    from wrist coordinates.

    # Arguments
        visibility_mask: Numpy array with shape `(42, 1)`.

    # Returns
        visibility_mask: Numpy array with shape `(42, 1)`.
    """
    visibility_left_root = visibility_mask[LEFT_WRIST]
    visibility_left_aligned = visibility_mask[LEFT_MIDDLE_METACARPAL]
    visibility_right_root = visibility_mask[RIGHT_WRIST]
    visibility_right_aligned = visibility_mask[RIGHT_MIDDLE_METACARPAL]

    palm_visibility_left = np.logical_or(
        visibility_left_root, visibility_left_aligned)
    palm_visibility_right = np.logical_or(
        visibility_right_root, visibility_right_aligned)

    palm_visibility_left = np.expand_dims(palm_visibility_left, 0)
    palm_visibility_right = np.expand_dims(palm_visibility_right, 0)

    visibility_mask = np.concatenate(
        [palm_visibility_left, visibility_mask[LEFT_WRIST: LEFT_PINKY_TIP],
         palm_visibility_right, visibility_mask[RIGHT_WRIST: RIGHT_PINKY_TIP]],
        0)
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
    palm_coordinates_left = 0.5 * (keypoints[LEFT_WRIST, :] +
                                   keypoints[LEFT_MIDDLE_METACARPAL, :])
    palm_coordinates_right = 0.5 * (keypoints[RIGHT_WRIST, :] +
                                    keypoints[RIGHT_MIDDLE_METACARPAL, :])

    palm_coordinates_left = np.expand_dims(palm_coordinates_left, 0)
    palm_coordinates_right = np.expand_dims(palm_coordinates_right, 0)

    keypoints = np.concatenate(
        [palm_coordinates_left, keypoints[LEFT_WRIST:LEFT_PINKY_TIP, :],
         palm_coordinates_right, keypoints[RIGHT_WRIST:RIGHT_PINKY_TIP, :]], 0)

    return keypoints


def get_transform_to_bone_frame(keypoints3D, bone_index):
    """ Transform the keypoints in camera image frame to index keypoint frame.

    # Arguments
        keypoints3D: numpy array of shape (num_keypoints, 3).
        bone_index: int value of range [0, num_keypoints].

    # Returns
        transformation_parameters: multiple values representing all the
        euclidean parameters to calculate transformation matrix.
    """
    index_keypoint = np.expand_dims(keypoints3D[bone_index, :], 1)
    translated_keypoint3D = to_homogeneous_coordinates(index_keypoint)
    translation_matrix = build_translation_matrix_SE3(np.zeros(3))
    translation_matrix = np.expand_dims(translation_matrix, 0)
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


def apply_root_transformations(keypoints3D, bone_index):
    """ Transform all keypoints to root keypoint frame.

    # Arguments
        keypoints3D: numpy array of shape (num_keypoints, 3).
        bone_index: int value of range [0, num_keypoints].

    # Returns
        relative_coordinates: numpy array of shape (num_keypoints, 3, 1).
        transformations: placeholder for transformation
        (num_keypoints, 4, 4, 1).
    """
    transformation_parameters = get_transform_to_bone_frame(keypoints3D,
                                                            bone_index)

    length_from_origin = transformation_parameters[0]
    rotation_angle_x = transformation_parameters[1]
    rotation_angle_y = transformation_parameters[2]
    rotated_keypoints = transformation_parameters[3]

    relative_coordinate = np.stack([length_from_origin, rotation_angle_x,
                                    rotation_angle_y], 0)
    return rotated_keypoints, relative_coordinate


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
    transformation_angles = get_transform_to_bone_frame(
        delta_vector, transformation_matrix)
    return transformation_angles


def apply_child_transformations(keypoints3D, bone_index, parent_index,
                                transformations):
    """ Calculate Child coordinate to Parent coordinate.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).
        bone_index: Index of current bone keypoint, Numpy array of shape (1).
        parent_index: Index of root keypoint, Numpy array of shape (1).
        relative_coordinates: place holder for relative_coordinates.
        transformations: placeholder for transformations.

    # Returns
        rotated_keypoints: place holder for relative_coordinates.
        transformation_parameters: placeholder for transformations.
    """
    transformation_matrix = transformations[parent_index]
    parent_keypoint_coordinates = transform_to_keypoint_coordinates(
        transformation_matrix, keypoints3D[parent_index, :])
    child_keypoint_coordinates = transform_to_keypoint_coordinates(
        transformation_matrix, keypoints3D[bone_index, :])
    transformation_parameters = get_articulation_angles(
        parent_keypoint_coordinates, child_keypoint_coordinates,
        transformation_matrix)
    length_from_origin = transformation_parameters[0]
    rotation_angle_x, rotation_angle_y = transformation_parameters[1:3]
    rotated_keypoints = transformation_parameters[3]
    transformation_parameters = np.stack([length_from_origin, rotation_angle_x,
                                          rotation_angle_y])
    return rotated_keypoints, transformation_parameters


def keypoints_to_root_frame(keypoints3D):
    """ Convert keypoints to root keypoint coordinates.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).

    # Returns
        relative_coordinates: keypoints in root keypoint coordinate frame.
    """
    transformations = [None] * len(KINEMATIC_CHAIN_LIST)
    relative_coordinates = np.zeros(len(KINEMATIC_CHAIN_LIST))
    for bone_index in KINEMATIC_CHAIN_LIST:
        parent_index = KINEMATIC_CHAIN_DICT[bone_index]
        if parent_index == 'root':
            transformation, relative_coordinate = apply_root_transformations(
                keypoints3D, bone_index)
        else:
            transformation, relative_coordinate = apply_child_transformations(
                keypoints3D, bone_index, parent_index, transformations)
        transformations[bone_index] = transformation
        relative_coordinates[bone_index] = relative_coordinate
    return relative_coordinates


def keypoint_to_root_frame(keypoints3D, num_keypoints=21):
    """ Convert keypoints to root keypoint coordinates.
    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).

    # Returns
        key_point_relative_frame: keypoints in root keypoint coordinate frame.
    """
    keypoints3D = keypoints3D.reshape([num_keypoints, 3])
    relative_coordinates = keypoints_to_root_frame(keypoints3D)
    key_point_relative_frame = np.stack(relative_coordinates, 1)
    key_point_relative_frame = np.squeeze(key_point_relative_frame)
    return key_point_relative_frame


def get_keypoints_z_rotation(keypoints3D, keypoint):
    """ Rotate Keypoints along z-axis.

    # Arguments
        keypoint: Keypoint to whose frame transformation is to
        be done, Numpy array of shape (1, 3).
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).

    # Returns
        reference_keypoint_z_rotation: Reference keypoint after rotation.
        resultant_keypoints3D: keypoints after rotation.
        rotation_matrix_z: Rotation matrix.
    """
    alpha = np.arctan2(keypoint[0], keypoint[1])
    rotation_matrix = build_rotation_matrix_z(alpha)
    keypoints3D = np.matmul(keypoints3D.T, rotation_matrix)
    keypoint = keypoints3D[LEFT_MIDDLE_METACARPAL, :]
    return keypoint, rotation_matrix, keypoints3D


def get_keypoints_x_rotation(keypoints3D, keypoint):
    """ Rotate Keypoints along x-axis.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).
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
    keypoint = keypoints3D[LEFT_PINKY_TIP, :]
    return keypoint, rotation_matrix, keypoints3D


def get_keypoints_y_rotation(keypoints3D, keypoint):
    """ Rotate Keypoints along y-axis.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).
        reference_keypoint: keypoint, Numpy array of shape (1, 3).

    # Returns
        resultant_keypoint: Resultant reference keypoint after rotation.
        resultant_keypoints3D: keypoints after rotation along Y-axis.
        rotation_matrix_y: Rotation matrix along x-axis.
    """
    gamma = np.arctan2(keypoint[2], keypoint[0])
    rotation_matrix = build_rotation_matrix_y(gamma)
    keypoints3D = np.matmul(keypoints3D, rotation_matrix)
    keypoint = keypoints3D[LEFT_PINKY_TIP, :]
    return keypoint, rotation_matrix, keypoints3D


def canonical_transformations_on_keypoints(keypoints3D):  # rename properly
    # RE_CHECK
    """ Transform Keypoints to canonical coordinates.

    # Arguments
        keypoints3D: Keypoints, Numpy array of shape (1, num_keypoints, 3).

    # Returns
        transformed_keypoints3D: Resultant keypoint after transformation.
        final_rotation_matrix: Final transformation matrix.
    """
    reference_keypoint = np.expand_dims(keypoints3D[:, LEFT_WRIST], 1)
    keypoints3D = keypoints3D - reference_keypoint
    keypoint = keypoints3D[:, LEFT_MIDDLE_METACARPAL]
    final_rotation_matrix = np.ones((3, 3))
    apply_rotations = [get_keypoints_z_rotation, get_keypoints_x_rotation,
                       get_keypoints_y_rotation]
    for function in apply_rotations:
        keypoint, rotation_matrix, keypoints3D = function(keypoints3D, keypoint)
        final_rotation_matrix = np.matmul(final_rotation_matrix,
                                          rotation_matrix)
    return np.squeeze(keypoints3D), np.squeeze(final_rotation_matrix)


def get_scale_matrix(scale, min_scale=1.0, max_scale=10.0):
    """ calculate scale matrix.

    # Arguments
        scale: Int value.

    # Returns
        scale_original: Int value
        scale_matrix: Numpy array of shape (3, 3)
    """
    scale_original = np.minimum(np.maximum(scale, min_scale), max_scale)
    scale_matrix = np.diag([scale_original, scale_original, 1])
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
    crop_size_halved = crop_size // 2
    translated_center_x = (crop_center[0] * scale) - crop_size_halved
    translated_center_y = (crop_center[1] * scale) - crop_size_halved
    translation_matrix = np.diag(
        [-translated_center_x, -translated_center_y, 1])
    return translation_matrix


def get_y_axis_rotated_keypoints(keypoint3D):
    """ Rotate keypoints along y-axis
    # Arguments
        keypoint3D: Numpy array of shape (num_keypoints, 3).

    # Returns
        keypoint3D: Numpy array of shape (num_keypoints, 3).
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
    """ Rotate keypoints along x-axis

    # Arguments
        keypoint3D: Numpy array of shape (num_keypoints, 3).
        length_from_origin: Numpy array of shape (1, ).
        rotation_matrix: Numpy array of shape (3, 3).

    # Returns
        keypoint3D: Numpy array of shape (num_keypoints, 3).
        affine_rotation_matrix_y: Numpy array of shape (3, 3).
        gamma: Numpy array of shape (1, ).
    """
    alpha = np.arctan2(-keypoint3D[1], keypoint3D[2])
    rotation_matrix_x = build_rotation_matrix_x(alpha)
    affine_rotation_matrix_x = build_affine_matrix(rotation_matrix_x)
    translation_matrix_to_origin = build_translation_matrix_SE3(
        -length_from_origin)
    translation_matrix_to_origin = np.expand_dims(translation_matrix_to_origin,
                                                  0)
    rotation_matrix_xy = np.matmul(affine_rotation_matrix_x, rotation_matrix)
    keypoint3D = np.matmul(translation_matrix_to_origin, rotation_matrix_xy)
    return keypoint3D, alpha


def get_transformation_parameters(keypoint3D, transformation_matrix):
    """ Calculate transformation parameters.

    # Arguments
        keypoint3D: Numpy array of shape (num_keypoints, 3).
        transformation_matrix: Numpy array of shape (4, 4).

    # Returns
        length_from_origin: float value.
        alpha: float value. Rotation angle along X-axis.
        gamma: float value. Rotation angle along X-axis.
        final_transformation_matrix: Numpy array of shape (4, 4).
    """
    length_from_origin = np.linalg.norm(keypoint3D)

    keypoint_parameters = get_y_axis_rotated_keypoints(keypoint3D)
    keypoint3D_rotated_y, affine_matrix, rotation_angle_y = keypoint_parameters

    keypoint3D_rotated_x, rotation_angle_x = get_x_axis_rotated_keypoints(
        keypoint3D_rotated_y, length_from_origin, affine_matrix)

    rotated_keypoints = np.matmul(keypoint3D_rotated_x, transformation_matrix)
    transformation_parameters = (length_from_origin, rotation_angle_x,
                                 rotation_angle_y, rotated_keypoints)

    return transformation_parameters


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
    cropped_keypoints[:, [0, 1]] = cropped_keypoints[:, [1, 0]]
    keypoints = np.copy(cropped_keypoints)
    keypoints = keypoints - (crop_size // 2)
    keypoints = keypoints / scale
    keypoints = keypoints + centers
    keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
    return keypoints


def canonical_to_relative_coordinates(num_keypoints, canonical_coordinates,
                                      rotation_matrix, hand_side):
    hand_arg = np.argmax(hand_side, 1)
    hand_side_mask = np.equal(hand_arg, 1)
    hand_side_mask = np.reshape(hand_side_mask, [-1, 1])
    hand_side_mask_3D = np.tile(hand_side_mask, [num_keypoints, 3])
    keypoint_flipped = flip_right_to_left_hand(canonical_coordinates,
                                               hand_side_mask_3D)
    relative_keypoints = np.matmul(keypoint_flipped, rotation_matrix)
    return relative_keypoints
