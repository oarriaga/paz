import tensorflow as tf
from tensorflow import keras
import numpy as np
from plyfile import PlyData
import math
from pose import LINEMOD_CAMERA_MATRIX


class MultiPoseLoss(object):
    def __init__(self, object_id, translation_priors, data_path,
                 target_num_points=500, num_pose_dims=3, model_path='models/',
                 translation_scale_norm=1000):
        self.object_id = object_id
        self.translation_priors = translation_priors
        self.num_pose_dims = num_pose_dims
        self.translation_scale_norm = translation_scale_norm
        self.model_path = data_path + model_path + 'obj_' + object_id + '.ply'
        self.model_points = self._load_model_file()
        self.model_points = self._filter_model_points(target_num_points)

    def _load_model_file(self):
        model_data = PlyData.read(self.model_path)
        vertex = model_data['vertex'][:]
        vertices = [vertex['x'], vertex['y'], vertex['z']]
        model_points = np.stack(vertices, axis=-1)
        return model_points

    def _filter_model_points(self, target_num_points):
        num_points = self.model_points.shape[0]

        if num_points == target_num_points:
            points = self.model_points
        elif num_points < target_num_points:
            points = np.zeros((target_num_points, 3))
            points[:num_points, :] = self.model_points
        else:
            step_size = (num_points // target_num_points) - 1
            step_size = max(1, step_size)
            points = self.model_points[::step_size, :]
            points = points[np.newaxis, :target_num_points, :]

        return tf.convert_to_tensor(points)

    def _separate_axis_from_angle(self, axis_angle):
        squared = tf.math.square(axis_angle)
        sum = tf.math.reduce_sum(squared, axis=-1)
        angle = tf.expand_dims(tf.math.sqrt(sum), axis=-1)
        axis = tf.math.divide_no_nan(axis_angle, angle)
        return axis, angle

    def _compute_tx_ty(self, translation_xy_Tz, camera_parameter):
        fx, fy = camera_parameter[0], camera_parameter[1],
        px, py = camera_parameter[2], camera_parameter[3],
        tz_scale, image_scale = camera_parameter[4], camera_parameter[5]

        x = translation_xy_Tz[:, 0] / image_scale
        y = translation_xy_Tz[:, 1] / image_scale
        tz = translation_xy_Tz[:, 2] * tz_scale

        x = x - px
        y = y - py

        tx = tf.math.multiply(x, tz) / fx
        ty = tf.math.multiply(y, tz) / fy

        tx = tf.expand_dims(tx, axis=0)
        ty = tf.expand_dims(ty, axis=0)
        tz = tf.expand_dims(tz, axis=0)

        translations = tf.concat([tx, ty, tz], axis=0)
        return tf.transpose(translations)

    def _regress_translation(self, translation_raw):
        stride = self.translation_priors[:, -1]
        x = self.translation_priors[:, 0] + (translation_raw[:, :, 0] * stride)
        y = self.translation_priors[:, 1] + (translation_raw[:, :, 1] * stride)
        Tz = translation_raw[:, :, 2]
        translations_predicted = tf.concat([x, y, Tz], axis=0)
        return tf.transpose(translations_predicted)

    def _compute_camera_parameter(self, image_scale, camera_matrix):
        camera_parameter = tf.convert_to_tensor([camera_matrix[0, 0],
                                                 camera_matrix[1, 1],
                                                 camera_matrix[0, 2],
                                                 camera_matrix[1, 2],
                                                 self.translation_scale_norm,
                                                 image_scale])
        return camera_parameter

    def _rotate(self, point, axis, angle, name=None):
        with tf.compat.v1.name_scope(name, "axis_angle_rotate",
                                     [point, axis, angle]):
            cos_angle = tf.cos(angle)
            axis_dot_point = self._dot(axis, point)
            return (point * cos_angle + self._cross(axis, point) *
                    tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle))

    def _dot(self, vector1, vector2, axis=-1, keepdims=True, name=None):
        with tf.compat.v1.name_scope(name, "vector_dot", [vector1, vector2]):
            return tf.reduce_sum(input_tensor=vector1 * vector2,
                                 axis=axis, keepdims=keepdims)

    def _cross(self, vector1, vector2, name=None):
        with tf.compat.v1.name_scope(name, "vector_cross", [vector1, vector2]):
            vector1_x = vector1[:, :, 0]
            vector1_y = vector1[:, :, 1]
            vector1_z = vector1[:, :, 2]
            vector2_x = vector2[:, :, 0]
            vector2_y = vector2[:, :, 1]
            vector2_z = vector2[:, :, 2]
            n_x = vector1_y * vector2_z - vector1_z * vector2_y
            n_y = vector1_z * vector2_x - vector1_x * vector2_z
            n_z = vector1_x * vector2_y - vector1_y * vector2_x
            return tf.stack((n_x, n_y, n_z), axis=-1)

    def _calc_sym_distances(self, sym_points_pred, sym_points_target):
        sym_points_pred = tf.expand_dims(sym_points_pred, axis=2)
        sym_points_target = tf.expand_dims(sym_points_target, axis=1)
        distances = tf.reduce_min(tf.norm(
            sym_points_pred - sym_points_target, axis=-1), axis=-1)
        return tf.reduce_mean(distances, axis=-1)

    def _calc_asym_distances(self, asym_points_pred, asym_points_target):
        distances = tf.norm(asym_points_pred - asym_points_target, axis=-1)
        return tf.reduce_mean(distances, axis=-1)

    def compute_loss(self, y_true, y_pred):
        rotation_pred = y_pred[:, :, :self.num_pose_dims]
        translation_pred = y_pred[:, :, self.num_pose_dims:]
        translation_pred = self._regress_translation(translation_pred)
        scale = y_true[0, 0, -1]
        camera_parameter = self._compute_camera_parameter(
            scale, LINEMOD_CAMERA_MATRIX)
        translation_pred = self._compute_tx_ty(translation_pred,
                                               camera_parameter)
        translation_pred = tf.expand_dims(translation_pred, axis=0)

        rotation_true = y_true[:, :, :self.num_pose_dims]
        translation_true = y_true[:, :, 2 * self.num_pose_dims:2 *
                                  self.num_pose_dims + self.num_pose_dims]
        is_symmetric = y_true[:, :, self.num_pose_dims]
        class_indices = y_true[:, :, self.num_pose_dims + 1]
        anchor_flags = y_true[:, :, -2]
        anchor_state = tf.cast(tf.math.round(anchor_flags), tf.int32)

        indices = tf.where(tf.equal(anchor_state, 1))
        rotation_pred = tf.gather_nd(rotation_pred, indices) * math.pi
        translation_pred = tf.gather_nd(translation_pred, indices)

        rotation_true = tf.gather_nd(rotation_true, indices) * math.pi
        translation_true = tf.gather_nd(translation_true, indices)

        is_symmetric = tf.gather_nd(is_symmetric, indices)
        is_symmetric = tf.cast(tf.math.round(is_symmetric), tf.int32)
        class_indices = tf.gather_nd(class_indices, indices)
        class_indices = tf.cast(tf.math.round(class_indices), tf.int32)

        axis_pred, angle_pred = self._separate_axis_from_angle(rotation_pred)
        axis_target, angle_target = self._separate_axis_from_angle(
            rotation_true)

        selected_model_points = tf.gather(self.model_points,
                                          class_indices, axis=0)
        axis_pred = tf.expand_dims(axis_pred, axis=1)
        angle_pred = tf.expand_dims(angle_pred, axis=1)
        axis_target = tf.expand_dims(axis_target, axis=1)
        angle_target = tf.expand_dims(angle_target, axis=1)

        translation_pred = tf.expand_dims(translation_pred, axis=1)
        translation_true = tf.expand_dims(translation_true, axis=1)

        transformed_points_pred = self._rotate(
            selected_model_points, axis_pred, angle_pred) + translation_pred
        transformed_points_target = (self._rotate(
            selected_model_points, axis_target, angle_target) +
            translation_true)

        sym_indices = tf.where(keras.backend.equal(is_symmetric, 1))
        asym_indices = tf.where(keras.backend.not_equal(is_symmetric, 1))

        num_points = selected_model_points.shape[1]
        sym_points_pred = tf.reshape(tf.gather_nd(
            transformed_points_pred, sym_indices), (-1, num_points, 3))
        asym_points_pred = tf.reshape(tf.gather_nd(
            transformed_points_pred, asym_indices), (-1, num_points, 3))

        sym_points_target = tf.reshape(tf.gather_nd(
            transformed_points_target, sym_indices), (-1, num_points, 3))
        asym_points_target = tf.reshape(tf.gather_nd(
            transformed_points_target, asym_indices), (-1, num_points, 3))

        sym_distances = self._calc_sym_distances(sym_points_pred,
                                                 sym_points_target)
        asym_distances = self._calc_asym_distances(asym_points_pred,
                                                   asym_points_target)

        distances = tf.concat([sym_distances, asym_distances], axis=0)
        loss = tf.math.reduce_mean(distances)
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        return loss
