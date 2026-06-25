import numpy as np
from keras import ops


# EfficientPose transformation loss (ADD/ADI on rotated 3D model points).
# y_true: (batch, num_boxes, 11) -> [0:3] axis-angle/pi, [3] is_symmetric,
#   [4] class, [6:9] translation, [-2] anchor flag, [-1] image scale.
# y_pred: (batch, num_boxes, 6)  -> [0:3] axis-angle/pi, [3:6] translation_raw.
# The legacy loss gathers the dynamic set of positive anchors; here we select a
# fixed number of top-flagged anchors per sample so the loss is jit-friendly,
# masking the padding with the gathered flag.
class MultiPoseLoss:
    def __init__(self, model_points, translation_priors, camera_matrix,
                 translation_scale_norm=1000.0, max_positives=64):
        self.model_points = ops.cast(model_points, "float32")
        self.translation_priors = ops.cast(translation_priors, "float32")
        self.camera_matrix = ops.cast(camera_matrix, "float32")
        self.tz_scale = float(translation_scale_norm)
        self.max_positives = max_positives

    def compute_loss(self, y_true, y_pred):
        flags = y_true[:, :, -2]
        scores, indices = ops.top_k(flags, self.max_positives)
        valid = ops.cast(scores > 0.5, "float32")
        rotation_true = gather_anchors(y_true[:, :, 0:3], indices)
        rotation_pred = gather_anchors(y_pred[:, :, 0:3], indices)
        translation_true = gather_anchors(y_true[:, :, 6:9], indices)
        translation_raw = gather_anchors(y_pred[:, :, 3:6], indices)
        is_symmetric = gather_anchors(y_true[:, :, 3:4], indices)[:, :, 0]
        priors = self.gather_priors(indices)
        scale = y_true[:, 0, -1]

        translation_pred = compute_translation(
            translation_raw, priors, scale, self.tz_scale, self.camera_matrix)
        points_true = transform_points(
            self.model_points, rotation_true * np.pi, translation_true)
        points_pred = transform_points(
            self.model_points, rotation_pred * np.pi, translation_pred)
        distances = compute_distances(points_true, points_pred, is_symmetric)
        return ops.sum(distances * valid) / ops.maximum(ops.sum(valid), 1.0)

    def gather_priors(self, indices):
        batch_size = ops.shape(indices)[0]
        priors = ops.broadcast_to(
            self.translation_priors[None], (batch_size, *self.translation_priors.shape))  # fmt: skip
        return gather_anchors(priors, indices)


def gather_anchors(tensor, indices):
    indices = ops.broadcast_to(
        indices[:, :, None], (*ops.shape(indices), ops.shape(tensor)[-1]))
    return ops.take_along_axis(tensor, indices, axis=1)


def transform_points(model_points, rotation, translation):
    axis, angle = separate_axis_from_angle(rotation)
    points = ops.broadcast_to(
        model_points[None, None], (*ops.shape(rotation)[:2], *model_points.shape))  # fmt: skip
    rotated = rotate(points, axis[:, :, None, :], angle[:, :, None, :])
    return rotated + translation[:, :, None, :]


def separate_axis_from_angle(axis_angle):
    angle = ops.sqrt(ops.sum(axis_angle ** 2, axis=-1, keepdims=True))
    axis = axis_angle / ops.maximum(angle, 1e-9)
    return axis, angle


def rotate(points, axis, angle):
    cos_angle = ops.cos(angle)
    axis_dot_point = ops.sum(axis * points, axis=-1, keepdims=True)
    return (points * cos_angle + cross(axis, points) * ops.sin(angle)
            + axis * axis_dot_point * (1.0 - cos_angle))


def cross(axis, points):
    axis_x, axis_y, axis_z = axis[..., 0], axis[..., 1], axis[..., 2]
    points_x, points_y, points_z = points[..., 0], points[..., 1], points[..., 2]  # fmt: skip
    cross_x = axis_y * points_z - axis_z * points_y
    cross_y = axis_z * points_x - axis_x * points_z
    cross_z = axis_x * points_y - axis_y * points_x
    return ops.stack([cross_x, cross_y, cross_z], axis=-1)


def compute_translation(translation_raw, priors, scale, tz_scale, camera):
    stride = priors[:, :, 2]
    x = priors[:, :, 0] + translation_raw[:, :, 0] * stride
    y = priors[:, :, 1] + translation_raw[:, :, 1] * stride
    z = translation_raw[:, :, 2] * tz_scale
    x = (x / scale[:, None] - camera[0, 2]) * z / camera[0, 0]
    y = (y / scale[:, None] - camera[1, 2]) * z / camera[1, 1]
    return ops.stack([x, y, z], axis=-1)


def compute_distances(points_true, points_pred, is_symmetric):
    asymmetric = ops.mean(ops.norm(points_pred - points_true, axis=-1), axis=-1)
    pairwise = points_pred[:, :, :, None] - points_true[:, :, None, :]
    nearest = ops.min(ops.norm(pairwise, axis=-1), axis=-1)
    symmetric = ops.mean(nearest, axis=-1)
    return ops.where(is_symmetric > 0.5, symmetric, asymmetric)
