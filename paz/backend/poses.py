import numpy as np
from .boxes import compute_ious, to_corner_form
from .mask import mask_to_box
from .image.opencv_image import (warp_affine, get_rotation_matrix,
                                 rotation_matrix_to_rotation_vector)


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


def rotation_matrix_to_axis_angle(rotations, num_pose_dims):
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
        rotation_vector = rotation_matrix_to_rotation_vector(rotation_matrix)
        axis_angle, jacobian = rotation_vector
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


def augment_pose_6D(image, boxes, rotation, translation_raw, mask,
                    scale_min, scale_max, angle_min, angle_max,
                    mask_value, input_size, camera_matrix):
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
        camera_matrix: Array with camera matrix of shape `(3, 3)`.

    # Returns:
        List: Containing augmented_image, augmented_boxes,
            augmented_rotation, augmented_translation, augmented_mask
    """
    transformation, angle, scale = generate_random_transformation(
        scale_min, scale_max, angle_min, angle_max, camera_matrix)
    augmented_image, augmented_mask = augment_images(transformation,
                                                     image, mask)
    (is_valid_augmentation, augmented_boxes, augmented_rotation,
     augmented_translation) = augment_annotations(
        boxes, scale, angle, rotation, translation_raw,
        augmented_mask, mask_value, input_size)

    if not is_valid_augmentation:
        augmented_image = image
        augmented_boxes = boxes
        augmented_rotation = rotation
        augmented_translation = translation_raw
        augmented_mask = mask
    return (augmented_image, augmented_boxes, augmented_rotation,
            augmented_translation, augmented_mask)


def generate_random_transformation(scale_min, scale_max, angle_min,
                                   angle_max, camera_matrix):
    """Generates random affine transformation matrix.

    # Arguments
        scale_min: Float, minimum value to scale image.
        scale_max: Float, maximum value to scale image.
        angle_min: Int, minimum degree to rotate image.
        angle_max: Int, maximum degree to rotate image.
        camera_matrix: Array with camera matrix of shape `(3, 3)`.

    # Returns:
        List: Containing transformation matrix, angle, scale
    """
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    angle = np.random.uniform(angle_min, angle_max)
    scale = np.random.uniform(scale_min, scale_max)
    return [get_rotation_matrix((cx, cy), -angle, scale), angle, scale]


def augment_images(transformation, image, mask):
    """Augments raw image and mask by applying affine transformation.

    # Arguments
        transformation: Array of shape (2,3) indicating affine transformation.
        image: Array raw image.
        mask: Array mask corresponding to raw image.

    # Returns:
        List: Containing augmented_image and augmented_mask.
    """
    H, W, _ = image.shape
    augmented_image = warp_affine(image, transformation, size=(W, H))
    H, W, _ = mask.shape
    augmented_mask = warp_affine(mask, transformation, size=(W, H))
    return [augmented_image, augmented_mask]


def augment_annotations(boxes, scale, angle, rotation, translation_raw,
                        augmented_mask, mask_value, input_size):
    """Augments pose data such as rotations and translations.
    and its corresponding poses.

    # Arguments
        boxes: Array of shape `(n, 5)`
        scale: Float, value to scale pose annotations.
        angle: Int, angle to rotate pose annotations.
        rotation: Array of shape `(n, 9)`
        translation_raw: Array of shape `(n, 3)`
        augmented_mask: Array mask corresponding to raw image.
        mask_value: Int, pixel gray value of foreground in mask image.
        input_size: Int, input image size of the model.

    # Returns:
        List: Containing is_valid_augmentation, augmented_boxes,
            augmented_rotation, and augmented_translation.
    """
    num_annotations = boxes.shape[0]
    augmented_boxes, is_valid = [], []
    rotation_vector = np.zeros((3, ))
    rotation_vector[2] = angle / 180 * np.pi
    transformation, _ = rotation_matrix_to_rotation_vector(rotation_vector)
    augmented_translation = np.empty_like(translation_raw)
    box = mask_to_box(augmented_mask, mask_value)
    rotation_matrices = np.reshape(rotation, (num_annotations, 3, 3))
    augmented_rotation = np.empty_like(rotation_matrices)
    is_valid_augmentation = sum(box)
    if is_valid_augmentation:
        for num_annotation in range(num_annotations):
            augmented_box = mask_to_box(augmented_mask, mask_value)
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
    return [is_valid_augmentation, augmented_boxes,
            augmented_rotation, augmented_translation]
