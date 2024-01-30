import cv2
import numpy as np
from .boxes import compute_ious, to_corner_form
from ..datasets import LINEMOD_CAMERA_MATRIX


def match_poses(boxes, poses, prior_boxes, iou_threshold):
    """Match prior boxes with poses with ground truth boxes and poses.

    # Arguments
        boxes: Array of shape `(n, 5)`.
        poses: Array of shape `(n, 5)`.
        prior_boxes: Array of shape `(num_boxes, 4)`.
        iou_threshold: Floats, IOU threshold value.

    # Returns
        matched_poses: Array of shape `(num_boxes, 6)`.
    """
    matched_poses = np.zeros((prior_boxes.shape[0], poses.shape[1] + 1))
    ious = compute_ious(boxes, to_corner_form(np.float32(prior_boxes)))
    per_prior_which_box_iou = np.max(ious, axis=0)
    per_prior_which_box_arg = np.argmax(ious, 0)
    per_box_which_prior_arg = np.argmax(ious, 1)
    per_prior_which_box_iou[per_box_which_prior_arg] = 2
    for box_arg in range(len(per_box_which_prior_arg)):
        best_prior_box_arg = per_box_which_prior_arg[box_arg]
        per_prior_which_box_arg[best_prior_box_arg] = box_arg
    matched_poses[:, :-1] = poses[per_prior_which_box_arg]
    matched_poses[per_prior_which_box_iou >= iou_threshold, -1] = 1
    return matched_poses


def transform_rotation(rotations, num_pose_dims):
    """Computes axis angle rotation vector from a rotation matrix.

    # Arguments:
        rotation: Array, of shape `(n, 9)`.
        num_pose_dims: Int, number of pose dimensions.

    # Returns:
        Array: of shape (n, 5) containing axis angle vector.
    """
    final_axis_angles = []
    for rotation in rotations:
        final_axis_angle = np.zeros((num_pose_dims + 2))
        rotation_matrix = np.reshape(rotation, (num_pose_dims, num_pose_dims))
        axis_angle, jacobian = cv2.Rodrigues(rotation_matrix)
        axis_angle = np.squeeze(axis_angle) / np.pi
        final_axis_angle[:3] = axis_angle
        final_axis_angle = np.expand_dims(final_axis_angle, axis=0)
        final_axis_angles.append(final_axis_angle)
    return np.concatenate(final_axis_angles, axis=0)


def concatenate_poses(rotations, translations):
    """Concatenates rotations and translations into a single array.

    # Arguments:
        rotations: Array, of shape `(num_boxes, 6)`.
        translations: Array, of shape `(num_boxes, 4)`.

    # Returns:
        Array: of shape (num_boxes, 10)
    """
    return np.concatenate((rotations, translations), axis=-1)


def concatenate_scale(poses, scale):
    """Concatenates poses and scale into a single array.

    # Arguments:
        poses: Array, of shape `(num_boxes, 10)`.
        scale: Array, of shape `()`.

    # Returns:
        Array: of shape (num_boxes, 11)
    """
    scale = np.repeat(scale, poses.shape[0])
    scale = scale[np.newaxis, :]
    return np.concatenate((poses, scale.T), axis=1)


def augment_6DOF(image, boxes, rotation, translation_raw, mask,
                 scale_min, scale_max, angle_min, angle_max,
                 mask_value, input_size):
    """Performs 6 degree of freedom augmentation of image
    and its corresponding poses.

    # Arguments
        image: Array raw image.
        boxes: Array of shape `(n, 5)`
        rotation: Array of shape `(n, 9)`
        translation_raw: Array of shape `(n, 3)`
        mask: Array mask corresponding to raw image.
        scale_min: Float, minimum value to scale image.
        scale_max: Float, maximum value to scale image.
        angle_min: Int, minimum degree to rotate image.
        angle_max: Int, maximum degree to rotate image.
        mask_value: Int, pixel gray value of foreground in mask image.
        input_size: Int, input image size of the model.

    # Returns:
        List: Containing augmented_image, augmented_boxes,
            augmented_rotation, augmented_translation, augmented_mask
    """
    transformation, angle, scale = generate_random_transformation(
        scale_min, scale_max, angle_min, angle_max)
    H, W, _ = image.shape
    augmented_image = cv2.warpAffine(image, transformation, (W, H),
                                     flags=cv2.INTER_CUBIC)
    H, W, _ = mask.shape
    augmented_mask = cv2.warpAffine(mask, transformation, (W, H),
                                    flags=cv2.INTER_NEAREST)
    num_annotations = boxes.shape[0]
    augmented_boxes, is_valid = [], []
    rotation_vector = np.zeros((3, ))
    rotation_vector[2] = angle / 180 * np.pi
    transformation, _ = cv2.Rodrigues(rotation_vector)
    augmented_translation = np.empty_like(translation_raw)
    box = compute_box_from_mask(augmented_mask, mask_value)
    rotation_matrices = np.reshape(rotation, (num_annotations, 3, 3))
    augmented_rotation = np.empty_like(rotation_matrices)
    is_valid_augmentation = sum(box)
    if is_valid_augmentation:
        for num_annotation in range(num_annotations):
            augmented_box = compute_box_from_mask(augmented_mask, mask_value)
            rotation_matrix = np.dot(transformation,
                                     rotation_matrices[num_annotation])
            translation_vector = np.dot(transformation,
                                        translation_raw[num_annotation].T)
            augmented_rotation[num_annotation] = rotation_matrix
            augmented_translation[num_annotation] = translation_vector
            augmented_translation[num_annotation][2] = augmented_translation[
                num_annotation][2] / scale
            augmented_boxes.append(augmented_box)
            is_valid.append(bool(sum(augmented_box)))
        augmented_boxes = np.array(augmented_boxes) / input_size
        augmented_boxes = np.concatenate((augmented_boxes, boxes[
            is_valid][:, -1][np.newaxis, :].T), axis=1)
        augmented_rotation = np.reshape(augmented_rotation,
                                        (num_annotations, 9))
    else:
        augmented_image = image
        augmented_boxes = boxes
        augmented_rotation = rotation
        augmented_translation = translation_raw
        augmented_mask = mask

    return (augmented_image, augmented_boxes, augmented_rotation,
            augmented_translation, augmented_mask)


def generate_random_transformation(scale_min, scale_max,
                                   angle_min, angle_max):
    """Generates random affine transformation matrix.

    # Arguments
        scale_min: Float, minimum value to scale image.
        scale_max: Float, maximum value to scale image.
        angle_min: Int, minimum degree to rotate image.
        angle_max: Int, maximum degree to rotate image.

    # Returns:
        List: Containing transformation matrix, angle, scale
    """
    cx = LINEMOD_CAMERA_MATRIX[0, 2]
    cy = LINEMOD_CAMERA_MATRIX[1, 2]
    angle = np.random.uniform(angle_min, angle_max)
    scale = np.random.uniform(scale_min, scale_max)
    return [cv2.getRotationMatrix2D((cx, cy), -angle, scale), angle, scale]


def compute_box_from_mask(mask, mask_value):
    """Computes bounding box from mask image.

    # Arguments
        mask: Array mask corresponding to raw image.
        mask_value: Int, pixel gray value of foreground in mask image.

    # Returns:
        box: List containing box coordinates.
    """
    masked = np.where(mask == mask_value)
    mask_x, mask_y = masked[1], masked[0]
    if mask_x.size <= 0 or mask_y.size <= 0:
        box = [0, 0, 0, 0]
    else:
        x_min, y_min = np.min(mask_x), np.min(mask_y)
        x_max, y_max = np.max(mask_x), np.max(mask_y)
        box = [x_min, y_min, x_max, y_max]
    return box