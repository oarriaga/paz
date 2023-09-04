import numpy as np
from paz.abstract import Processor, Pose6D
import paz.processors as pr
from paz.processors.draw import (quaternion_to_rotation_matrix,
                                 project_to_image, draw_cube)


class ComputeResizingShape(Processor):
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
    image = np.pad(image, pad_shape, mode=mode)
    return image


class ComputeCameraParameter(Processor):
    def __init__(self, camera_matrix, translation_scale_norm):
        self.camera_matrix = camera_matrix
        self.translation_scale_norm = translation_scale_norm
        super(ComputeCameraParameter, self).__init__()

    def call(self, image_scale):
        return compute_camera_parameter(image_scale, self.camera_matrix,
                                        self.translation_scale_norm)


def compute_camera_parameter(image_scale, camera_matrix,
                             translation_scale_norm):
    camera_parameter = np.array([camera_matrix[0, 0],
                                 camera_matrix[1, 1],
                                 camera_matrix[0, 2],
                                 camera_matrix[1, 2],
                                 translation_scale_norm,
                                 image_scale])
    return camera_parameter


class RegressTranslation(Processor):
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
    translations_predicted = np.concatenate((x, y, Tz), axis=0)
    return translations_predicted.T


class ComputeTxTy(Processor):
    def __init__(self):
        super(ComputeTxTy, self).__init__()

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

    translations = np.concatenate((tx, ty, tz), axis=0)
    return translations.T


class ComputeSelectedIndices(Processor):
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
    """Transforms boxes from dataset into `Boxes2D` messages.

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
    """Transforms boxes from dataset into `Boxes2D` messages given boxes
    with scores as one hot vectors.

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
    """Transforms boxes from dataset into `Boxes2D` messages given no
    class names and score.

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
    """Transforms boxes from dataset into `Boxes2D` messages given boxes
    with class argument.

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
    """Draws cube in image by projecting points3D with intrinsics and pose6D.

    # Arguments
        image: Array (H, W).
        pose6D: paz.abstract.Pose6D instance.
        intrinsics: Array (3, 3). Camera intrinsics for projecting
            3D rays into 2D image.
        points3D: Array (num_points, 3).
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    quaternion, translation = pose6D.quaternion, pose6D.translation
    rotation = quaternion_to_rotation_matrix(quaternion)
    points2D = project_to_image(rotation, translation, points3D, intrinsics)
    image = draw_cube(image, points2D.astype(np.int32),
                      thickness=thickness, color=color)
    return image
