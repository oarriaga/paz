import numpy as np
import cv2
from paz.abstract import Processor, SequentialProcessor, Pose6D
import paz.processors as pr
from paz.processors.draw import (quaternion_to_rotation_matrix,
                                 project_to_image, draw_cube)
from paz.backend.boxes import compute_ious, to_corner_form

LINEMOD_CAMERA_MATRIX = np.array([
    [572.41140, 000.00000, 325.26110],
    [000.00000, 573.57043, 242.04899],
    [000.00000, 000.00000, 001.00000]],
    dtype=np.float32)


class ComputeResizingShape(Processor):
    """Computes the final size of the image to be scaled by `size`
    such that the maximum dimension of the image is equal to `size`.

    # Arguments
        size: Int, final size of maximum dimension of the image.
    """
    def __init__(self, size):
        self.size = size
        super(ComputeResizingShape, self).__init__()

    def call(self, image):
        return compute_resizing_shape(image, self.size)


def compute_resizing_shape(image, size):
    H, W = image.shape[:2]
    image_scale = size / max(H, W)
    resizing_W = int(W * image_scale)
    resizing_H = int(H * image_scale)
    resizing_shape = (resizing_W, resizing_H)
    return resizing_shape, np.array(image_scale)


class PadImage(Processor):
    """Pads the image to the final size `size`.

    # Arguments
        size: Int, final size of maximum dimension of the image.
        mode: Str, specifying the type of padding.
    """
    def __init__(self, size, mode='constant'):
        self.size = size
        self.mode = mode
        super(PadImage, self).__init__()

    def call(self, image):
        return pad_image(image, self.size, self.mode)


def pad_image(image, size, mode):
    H, W = image.shape[:2]
    pad_H = size - H
    pad_W = size - W
    pad_shape = [(0, pad_H), (0, pad_W), (0, 0)]
    return np.pad(image, pad_shape, mode=mode)


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
    return np.array([camera_matrix[0, 0], camera_matrix[1, 1],
                     camera_matrix[0, 2], camera_matrix[1, 2],
                     translation_scale_norm, image_scale])


class RegressTranslation(Processor):
    """Applies regression offset values to translation
    anchors to get the 2D translation center-point and Tz.

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
        return compute_tx_ty(translation_xy_Tz, camera_parameter)


def compute_tx_ty(translation_xy_Tz, camera_parameter):
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
        class_names: List of class names ordered with respect to the
            class indices from the dataset ``boxes``.
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
        intrinsics: Array (3, 3). Camera intrinsics for projecting
            3D rays into 2D image.
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
        transformed_rotations: Array of shape (5,) containing the
            transformed rotation.
    """
    def __init__(self, num_pose_dims):
        self.num_pose_dims = num_pose_dims
        super(TransformRotation, self).__init__()

    def call(self, rotations):
        return transform_rotation(rotations, self.num_pose_dims)


def transform_rotation(rotations, num_pose_dims):
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
        poses_combined: Array of shape `(num_prior_boxes, 10)`
            containing the transformed rotation.
    """
    def __init__(self):
        super(ConcatenatePoses, self).__init__()

    def call(self, rotations, translations):
        return concatenate_poses(rotations, translations)


def concatenate_poses(rotations, translations):
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
    transformation, angle, scale = generate_random_transformation(
        scale_min, scale_max, angle_min, angle_max)
    augmented_image = apply_transformation(
        image, transformation, cv2.INTER_CUBIC)
    augmented_mask = apply_transformation(
        mask, transformation, cv2.INTER_NEAREST)

    num_annotations = boxes.shape[0]
    augmented_boxes, is_valid = [], []
    rotation_vector = compute_rotation_vector(angle)
    transformation, _ = cv2.Rodrigues(rotation_vector)
    augmented_translation = np.empty_like(translation_raw)
    box = compute_box_from_mask(augmented_mask, mask_value)
    rotation_matrices = np.reshape(rotation, (num_annotations, 3, 3))
    augmented_rotation = np.empty_like(rotation_matrices)
    is_valid_augmentation = sum(box)
    if is_valid_augmentation:
        for num_annotation in range(num_annotations):
            augmented_box = compute_box_from_mask(augmented_mask, mask_value)
            rotation_matrix = transform_rotation_matrix(
                rotation_matrices[num_annotation], transformation)
            translation_vector = transform_translation_vector(
                translation_raw[num_annotation], transformation)
            augmented_rotation[num_annotation] = rotation_matrix
            augmented_translation[num_annotation] = translation_vector
            augmented_translation[num_annotation] = scale_translation_vector(
                augmented_translation[num_annotation], scale)
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
    cx = LINEMOD_CAMERA_MATRIX[0, 2]
    cy = LINEMOD_CAMERA_MATRIX[1, 2]
    angle = np.random.uniform(angle_min, angle_max)
    scale = np.random.uniform(scale_min, scale_max)
    return [cv2.getRotationMatrix2D((cx, cy), -angle, scale), angle, scale]


def apply_transformation(image, transformation, interpolation):
    H, W, _ = image.shape
    return cv2.warpAffine(image, transformation, (W, H), flags=interpolation)


def compute_box_from_mask(mask, mask_value):
    segmentation = np.where(mask == mask_value)
    segmentation_x, segmentation_y = segmentation[1], segmentation[0]
    if segmentation_x.size <= 0 or segmentation_y.size <= 0:
        box = [0, 0, 0, 0]
    else:
        x_min, y_min = np.min(segmentation_x), np.min(segmentation_y)
        x_max, y_max = np.max(segmentation_x), np.max(segmentation_y)
        box = [x_min, y_min, x_max, y_max]
    return box


def compute_rotation_vector(angle):
    rotation_vector = np.zeros((3, ))
    rotation_vector[2] = angle / 180 * np.pi
    return rotation_vector


def transform_rotation_matrix(rotation_matrix, transformation):
    return np.dot(transformation, rotation_matrix)


def transform_translation_vector(translation, transformation):
    return np.dot(transformation, translation.T)


def scale_translation_vector(translation, scale):
    translation[2] = translation[2] / scale
    return translation


class AugmentColorspace(SequentialProcessor):
    def __init__(self):
        super(AugmentColorspace, self).__init__()
        self.add(AutoContrast())
        self.add(EqualizeHistogram())
        self.add(InvertColors())
        self.add(Posterize())
        self.add(Solarize())
        self.add(pr.RandomSaturation(1.0))
        self.add(pr.RandomContrast(1.0))
        self.add(pr.RandomBrightness(1.0))
        self.add(SharpenImage())
        self.add(Cutout())
        self.add(pr.RandomImageBlur())
        self.add(pr.RandomGaussianBlur())
        self.add(AddGaussianNoise())


class AutoContrast(Processor):
    def __init__(self, probability=0.50):
        self.probability = probability
        super(AutoContrast, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = auto_contrast(image)
        return image


def auto_contrast(image):
    contrasted = np.empty_like(image)
    num_channels = image.shape[2]

    for channel_arg in range(num_channels):
        image_per_channel = image[:, :, channel_arg]
        histogram = cv2.calcHist(image_per_channel, [0], None, [256], [0, 256])
        histogram[0] = -histogram[0]

        CDF_reversed = np.cumsum(histogram[::-1])
        upper_cutoff_nonzero = np.nonzero(CDF_reversed > 0.0)[0]
        if len(upper_cutoff_nonzero) == 0:
            upper_cutoff = -1
        else:
            upper_cutoff = 255 - upper_cutoff_nonzero[0]

        histogram[upper_cutoff+1:] = 0
        if upper_cutoff > -1:
            histogram[upper_cutoff] = CDF_reversed[255-upper_cutoff]

        for lower_cutoff, lower_value in enumerate(histogram):
            if lower_value:
                break

        if upper_cutoff <= lower_cutoff:
            lut = np.arange(256)
        else:
            scale = 255.0 / (upper_cutoff - lower_cutoff)
            offset = -lower_cutoff * scale
            lut = np.arange(256).astype(np.float64) * scale + offset
            lut = np.clip(lut, 0, 255).astype(np.uint8)
        lut = np.array(lut, dtype=np.uint8)
        contrast_adjusted = cv2.LUT(image_per_channel, lut)
        contrasted[:, :, channel_arg] = contrast_adjusted
    return contrasted


class EqualizeHistogram(Processor):
    """The paper uses Histogram euqlaization algorithm from PIL.
    This version of Histogram equalization produces slightly different
    results from that in the paper.
    """
    def __init__(self, probability=0.50):
        self.probability = probability
        super(EqualizeHistogram, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = equalize_histogram(image)
        return image


def equalize_histogram(image):
    equalized = np.empty_like(image)
    num_channels = image.shape[2]

    for channel_arg in range(num_channels):
        image_per_channel = image[:, :, channel_arg]
        equalized_per_channel = cv2.equalizeHist(image_per_channel)
        equalized[:, :, channel_arg] = equalized_per_channel
    return equalized


class InvertColors(Processor):
    def __init__(self, probability=0.50):
        self.probability = probability
        super(InvertColors, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = invert_colors(image)
        return image


def invert_colors(image):
    image_inverted = 255 - image
    return image_inverted


class Posterize(Processor):
    def __init__(self, probability=0.50, num_bits=4):
        self.probability = probability
        self.num_bits = num_bits
        super(Posterize, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = posterize(image, self.num_bits)
        return image


def posterize(image, num_bits):
    posterized = np.empty_like(image)
    num_channels = image.shape[2]
    for channel_arg in range(num_channels):
        image_per_channel = image[:, :, channel_arg]
        scale_factor = 2 ** (8 - num_bits)
        posterized_per_channel = np.round(image_per_channel /
                                          scale_factor) * scale_factor
        posterized[:, :, channel_arg] = posterized_per_channel.astype(np.uint8)
    return posterized


class Solarize(Processor):
    def __init__(self, probability=0.50, threshold=225):
        self.probability = probability
        self.threshold = threshold
        super(Solarize, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = solarize(image, self.threshold)
        return image


def solarize(image, threshold):
    return np.where(image < threshold, image, 255 - image)


class SharpenImage(Processor):
    def __init__(self, probability=0.50):
        self.probability = probability
        self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        super(SharpenImage, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = sharpen_image(image, self.kernel)
        return image


def sharpen_image(image, kernel):
    return cv2.filter2D(image, -1, kernel)


class Cutout(Processor):
    def __init__(self, probability=0.50, size=16, fill=128):
        self.probability = probability
        self.size = size
        self.fill = fill
        super(Cutout, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = cutout(image, self.size, self.fill)
        return image


def cutout(image, size, fill):
    H, W, _ = image.shape
    y = np.random.randint(0, H - size)
    x = np.random.randint(0, W - size)
    image[y:y+size, x:x+size, :] = fill
    return image


class AddGaussianNoise(Processor):
    def __init__(self, probability=0.50, mean=0, scale=20):
        self.probability = probability
        self.mean = mean
        self.variance = (scale / 100.0) * 255
        self.sigma = self.variance ** 0.5
        super(AddGaussianNoise, self).__init__()

    def call(self, image):
        if self.probability > np.random.rand():
            image = add_gaussian_noise(image, self.mean, self.sigma)
        return image


def add_gaussian_noise(image, mean, sigma):
    H, W, num_channels = image.shape
    noise = np.random.normal(mean, sigma, (H, W, num_channels))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)
