import numpy as np
import cv2
from paz.abstract import Processor, Pose6D
import paz.processors as pr
from paz.processors.draw import (quaternion_to_rotation_matrix,
                                 project_to_image, draw_cube)
from paz.backend.boxes import compute_ious, to_corner_form

LINEMOD_CAMERA_MATRIX = np.array([
    [572.41140, 000.00000, 325.26110],
    [000.00000, 573.57043, 242.04899],
    [000.00000, 000.00000, 001.00000]],
    dtype=np.float32)


class ComputeCameraParameter(Processor):
    """Computes camera parameter given camera matrix
    and scale normalization factor of translation.

    # Arguments
        camera_matrix: Array of shape `(3, 3)` camera matrix.
        translation_scale_norm: Float, factor to change units.
            EfficientPose internally works with meter and if the
            dataset unit is mm for example, then this parameter
            should be set to 1000.
    """
    def __init__(self, camera_matrix, translation_scale_norm):
        self.camera_matrix = camera_matrix
        self.translation_scale_norm = translation_scale_norm
        super(ComputeCameraParameter, self).__init__()

    def call(self, image_scale):
        return compute_camera_parameter(image_scale, self.camera_matrix,
                                        self.translation_scale_norm)


def compute_camera_parameter(image_scale, camera_matrix,
                             translation_scale_norm):
    """Computes camera parameter given camera matrix
    and scale normalization factor of translation.

    # Arguments
        image_scale: Array, scale of image.
        camera_matrix: Array, Camera matrix.
        translation_scale_norm: Float, factor to change units.
            EfficientPose internally works with meter and if the
            dataset unit is mm for example, then this parameter
            should be set to 1000.

    # Returns
        Array: of shape `(6,)` Camera parameter.
    """
    return np.array([camera_matrix[0, 0], camera_matrix[1, 1],
                     camera_matrix[0, 2], camera_matrix[1, 2],
                     translation_scale_norm, image_scale])


class RegressTranslation(Processor):
    """Applies regression offset values to translation anchors
    to get the 2D translation center-point and Tz.

    # Arguments
        translation_priors: Array of shape `(num_boxes, 3)`,
            translation anchors.
    """
    def __init__(self, translation_priors):
        self.translation_priors = translation_priors
        super(RegressTranslation, self).__init__()

    def call(self, translation_raw):
        return regress_translation(translation_raw, self.translation_priors)


def regress_translation(translation_raw, translation_priors):
    """Applies regression offset values to translation anchors
    to get the 2D translation center-point and Tz.

    # Arguments
        translation_raw: Array of shape `(1, num_boxes, 3)`,
        translation_priors: Array of shape `(num_boxes, 3)`,
            translation anchors.

    # Returns
        Array: of shape `(num_boxes, 3)`.
    """
    stride = translation_priors[:, -1]
    x = translation_priors[:, 0] + (translation_raw[:, :, 0] * stride)
    y = translation_priors[:, 1] + (translation_raw[:, :, 1] * stride)
    Tz = translation_raw[:, :, 2]
    return np.concatenate((x, y, Tz), axis=0).T


class ComputeTxTyTz(Processor):
    """Computes the Tx and Ty components of the translation vector
    with a given 2D-point and the intrinsic camera parameters.
    """
    def __init__(self):
        super(ComputeTxTyTz, self).__init__()

    def call(self, translation_xy_Tz, camera_parameter):
        return compute_tx_ty_tz(translation_xy_Tz, camera_parameter)


def compute_tx_ty_tz(translation_xy_Tz, camera_parameter):
    """Computes Tx, Ty and Tz components of the translation vector
    with a given 2D-point and the intrinsic camera parameters.

    # Arguments
        translation_xy_Tz: Array of shape `(num_boxes, 3)`,
        camera_parameter: Array: of shape `(6,)` camera parameter.

    # Returns
        Array: of shape `(num_boxes, 3)`.
    """
    fx, fy = camera_parameter[0], camera_parameter[1],
    px, py = camera_parameter[2], camera_parameter[3],
    tz_scale, image_scale = camera_parameter[4], camera_parameter[5]

    x = translation_xy_Tz[:, 0] / image_scale
    y = translation_xy_Tz[:, 1] / image_scale
    tz = translation_xy_Tz[:, 2] * tz_scale

    x = x - px
    y = y - py
    tx = np.multiply(x, tz) / fx
    ty = np.multiply(y, tz) / fy
    tx, ty, tz = tx[np.newaxis, :], ty[np.newaxis, :], tz[np.newaxis, :]
    return np.concatenate((tx, ty, tz), axis=0).T


class ComputeSelectedIndices(Processor):
    """Computes row-wise intersection between two given
    arrays and returns the indices of the intersections.
    """
    def __init__(self):
        super(ComputeSelectedIndices, self).__init__()

    def call(self, box_data_raw, box_data):
        return compute_selected_indices(box_data_raw, box_data)


def compute_selected_indices(box_data_all, box_data):
    """Computes row-wise intersection between two given
    arrays and returns the indices of the intersections.

    # Arguments
        box_data_all: Array of shape `(num_boxes, 5)`,
        box_data: Array: of shape `(n, 5)` box data.

    # Returns
        Array: of shape `(n, 3)`.
    """
    box_data_all_tuple = [tuple(row) for row in box_data_all[:, :4]]
    box_data_tuple = [tuple(row) for row in box_data[:, :4]]
    location_indices = []
    for tuple_element in box_data_tuple:
        location_index = box_data_all_tuple.index(tuple_element)
        location_indices.append(location_index)
    return np.array(location_indices)


class ToPose6D(Processor):
    """Transforms poses i.e rotations and
    translations into `Pose6D` messages.

    # Arguments
        class_names: List of class names ordered with respect
            to the class indices from the dataset ``boxes``.
        one_hot_encoded: Bool, indicating if scores are one hot vectors.
        default_score: Float, score to set.
        default_class: Str, class to set.
        box_method: Int, method to convert boxes to ``Boxes2D``.

    # Properties
        one_hot_encoded: Bool.
        box_processor: Callable.

    # Methods
        call()
    """
    def __init__(
            self, class_names=None, one_hot_encoded=False,
            default_score=1.0, default_class=None, box_method=0):
        if class_names is not None:
            arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        method_to_processor = {
            0: BoxesWithOneHotVectorsToPose6D(arg_to_class),
            1: BoxesToPose6D(default_score, default_class),
            2: BoxesWithClassArgToPose6D(arg_to_class, default_score)}
        self.pose_processor = method_to_processor[box_method]
        super(ToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        return self.pose_processor(box_data, rotations, translations)


class BoxesWithOneHotVectorsToPose6D(Processor):
    """Transforms poses into `Pose6D` messages
    given boxes with scores as one hot vectors.

    # Arguments
        arg_to_class: List, of classes.

    # Properties
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class):
        self.arg_to_class = arg_to_class
        super(BoxesWithOneHotVectorsToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        poses6D = []
        for box, rotation, translation in zip(box_data, rotations,
                                              translations):
            class_scores = box[4:]
            class_arg = np.argmax(class_scores)
            class_name = self.arg_to_class[class_arg]
            poses6D.append(Pose6D.from_rotation_vector(rotation, translation,
                                                       class_name))
        return poses6D


class BoxesToPose6D(Processor):
    """Transforms poses into `Pose6D` messages
    given no class names and score.

    # Arguments
        default_score: Float, score to set.
        default_class: Str, class to set.

    # Properties
        default_score: Float.
        default_class: Str.

    # Methods
        call()
    """
    def __init__(self, default_score=1.0, default_class=None):
        self.default_score = default_score
        self.default_class = default_class
        super(BoxesToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        poses6D = []
        for box, rotation, translation in zip(box_data, rotations,
                                              translations):
            poses6D.append(Pose6D.from_rotation_vector(rotation, translation,
                                                       self.default_class))
        return poses6D


class BoxesWithClassArgToPose6D(Processor):
    """Transforms poses into `Pose6D` messages
    given boxes with class argument.

    # Arguments
        default_score: Float, score to set.
        arg_to_class: List, of classes.

    # Properties
        default_score: Float.
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class, default_score=1.0):
        self.default_score = default_score
        self.arg_to_class = arg_to_class
        super(BoxesWithClassArgToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        poses6D = []
        for box, rotation, translation in zip(box_data, rotations,
                                              translations):
            class_name = self.arg_to_class[box[-1]]
            poses6D.append(Pose6D.from_rotation_vector(rotation, translation,
                                                       class_name))
        return poses6D


class DrawPose6D(pr.DrawPose6D):
    """Draws 3D bounding boxes from Pose6D messages.

    # Arguments
        object_sizes:  Array, of shape `(3,)` size of the object.
        camera_intrinsics: Array of shape `(3, 3)`,
            inrtrinsic camera parameter.
        box_color: List, the color to draw 3D bounding boxes.
    """
    def __init__(self, object_sizes, camera_intrinsics, box_color):
        self.box_color = box_color
        super().__init__(object_sizes, camera_intrinsics)

    def call(self, image, pose6D):
        if pose6D is None:
            return image
        image = draw_pose6D(image, pose6D, self.points3D, self.intrinsics,
                            self.thickness, self.box_color)
        return image


def draw_pose6D(image, pose6D, points3D, intrinsics, thickness, color):
    """Draws cube in image by projecting points3D with intrinsics
    and pose6D.

    # Arguments
        image: Array (H, W).
        pose6D: paz.abstract.Pose6D instance.
        intrinsics: Array (3, 3). Camera intrinsics
            for projecting 3D rays into 2D image.
        points3D: Array (num_points, 3).
        thickness: Positive integer indicating line thickness.
        color: List, the color to draw 3D bounding boxes.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    quaternion, translation = pose6D.quaternion, pose6D.translation
    rotation = quaternion_to_rotation_matrix(quaternion)
    points2D = project_to_image(rotation, translation, points3D, intrinsics)
    image = draw_cube(image, points2D.astype(np.int32),
                      thickness=thickness, color=color)
    return image


class MatchPoses(Processor):
    """Match prior boxes with ground truth poses.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        iou: Float in [0, 1]. Intersection over union in which prior
            boxes will be considered positive. A positive box is box
            with a class different than `background`.
    """
    def __init__(self, prior_boxes, iou=.5):
        self.prior_boxes = prior_boxes
        self.iou = iou
        super(MatchPoses, self).__init__()

    def call(self, boxes, poses):
        return match_poses(boxes, poses, self.prior_boxes, self.iou)


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


class TransformRotation(Processor):
    """Computes axis angle rotation vector from a rotation matrix.

    # Arguments:
        num_pose_dims: Int, number of dimensions of pose.

    # Returns:
        transformed_rotations: Array of shape (5,)
            containing transformed rotation.
    """
    def __init__(self, num_pose_dims):
        self.num_pose_dims = num_pose_dims
        super(TransformRotation, self).__init__()

    def call(self, rotations):
        return transform_rotation(rotations, self.num_pose_dims)


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


class ConcatenatePoses(Processor):
    """Concatenates rotations and translations into a single array.

    # Returns:
        poses_combined: Array of shape `(num_boxes, 10)`
            containing the transformed rotation.
    """
    def __init__(self):
        super(ConcatenatePoses, self).__init__()

    def call(self, rotations, translations):
        return concatenate_poses(rotations, translations)


def concatenate_poses(rotations, translations):
    """Concatenates rotations and translations into a single array.

    # Arguments:
        rotations: Array, of shape `(num_boxes, 6)`.
        translations: Array, of shape `(num_boxes, 4)`.

    # Returns:
        Array: of shape (num_boxes, 10)
    """
    return np.concatenate((rotations, translations), axis=-1)


class ConcatenateScale(Processor):
    """Concatenates poses with image scale into a single array.

    # Returns:
        poses_combined: Array of shape `(num_prior_boxes, 11)`
            containing the transformed rotation.
    """
    def __init__(self):
        super(ConcatenateScale, self).__init__()

    def call(self, poses, scale):
        return concatenate_scale(poses, scale)


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


class ScaleBoxes2D(Processor):
    """Scales coordinates of Boxes2D.

    # Returns:
        boxes2D: List, containg Boxes2D with scaled coordinates.
    """
    def __init__(self):
        super(ScaleBoxes2D, self).__init__()

    def call(self, boxes2D, scale):
        return scale_boxes2D(boxes2D, scale)


def scale_boxes2D(boxes2D, scale):
    """Scales coordinates of Boxes2D.

    # Arguments:
        boxes2D: List, of Box2D objects.
        scale: Foat, scale value.

    # Returns:
        boxes2D: List, of Box2D objects with scale coordinates.
    """
    for box2D in boxes2D:
        box2D.coordinates = tuple(np.array(box2D.coordinates) * scale)
    return boxes2D


class Augment6DOF(Processor):
    """Augment images, boxes, rotation and translation vector
    for pose estimation.

    # Arguments
        scale_min: Float, minimum value to scale image.
        scale_max: Float, maximum value to scale image.
        angle_min: Int, minimum degree to rotate image.
        angle_max: Int, maximum degree to rotate image.
        probability: Float, probability of data transformation.
        mask_value: Int, pixel gray value of foreground in mask image.
        input_size: Int, input image size of the model.
    """
    def __init__(self, scale_min=0.7, scale_max=1.3, angle_min=0,
                 angle_max=360, probability=0.5, mask_value=255,
                 input_size=512):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.probability = probability
        self.mask_value = mask_value
        self.input_size = input_size
        super(Augment6DOF, self).__init__()

    def call(self, image, boxes, rotation, translation_raw, mask):
        if np.random.rand() < self.probability:
            augmented_data = augment_6DOF(
                image, boxes, rotation, translation_raw, mask,
                self.scale_min, self.scale_max, self.angle_min,
                self.angle_max, self.mask_value, self.input_size)
        else:
            augmented_data = image, boxes, rotation, translation_raw, mask
        return augmented_data


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
