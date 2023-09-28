import tensorflow as tf
from tensorflow import keras
import numpy as np
from plyfile import PlyData
import natsort
import math
import os
from pose import LINEMOD_CAMERA_MATRIX


class MultiTransformationLoss(object):
    """Multi-box loss for a single-shot detection architecture.

    # Arguments
        neg_pos_ratio: Int. Number of negatives used per positive box.
        alpha: Float. Weight parameter for localization loss.
        max_num_negatives: Int. Maximum number of negatives per batch.

    # References
        - [SSD: Single Shot MultiBox
            Detector](https://arxiv.org/abs/1512.02325)
    """
    def __init__(self, translation_priors, neg_pos_ratio=3,
                 alpha=1.0, max_num_negatives=300):
        self.translation_priors = translation_priors
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.max_num_negatives = max_num_negatives
        self.object_id = '08'
        self.model_path = 'Linemod_preprocessed/models/'
        self.model_files = self.list_model_files()
        self.object_models = self.load_model_points()
        self.model_3d_points = self.get_model_3d_points()
        self.model_3d_points_for_loss = self.get_model_3d_points_for_loss(500)

    def list_model_files(self):
        all_files = os.listdir(self.model_path)
        model_files = [file for file in all_files if file.endswith('.ply')]
        model_files = natsort.natsorted(model_files)        
        return model_files

    def load_model_points(self):
        object_id_to_points = {}
        for model_file in self.model_files:
            full_model_file = self.model_path + model_file
            model_data = PlyData.read(full_model_file)
            vertex = model_data['vertex']
            points_3d = np.stack(
                [vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis=-1)
            object_id = model_file.split('.')[0][-2:]
            object_id_to_points[object_id] = points_3d
        return object_id_to_points

    def get_model_3d_points(self):
        return self.object_models[self.object_id]

    def get_model_3d_points_for_loss(self, points_for_shape_match_loss,
                                     flatten=False):
        num_points = self.model_3d_points.shape[0]

        if num_points == points_for_shape_match_loss:
            if flatten:
                to_return = np.reshape(self.model_3d_points, (-1,))
                return to_return
            else:
                return self.model_3d_points
        elif num_points < points_for_shape_match_loss:
            points = np.zeros((points_for_shape_match_loss, 3))
            points[:num_points, :] = self.model_3d_points
            if flatten:
                to_return = np.reshape(points, (-1,))
                return to_return
            else:
                return points
        else:
            step_size = (num_points // points_for_shape_match_loss) - 1
            if step_size < 1:
                step_size = 1
            points = self.model_3d_points[::step_size, :]
            if flatten:
                to_return = np.reshape(points[:points_for_shape_match_loss, :], (-1, ))
                return to_return
            else:
                to_return = points[np.newaxis, :points_for_shape_match_loss, :]
                to_return = tf.convert_to_tensor(value=to_return)
                return to_return

    def compute_loss(self, y_true, y_pred):
        """Computes localization and classification losses in a batch.

        # Arguments
            y_true: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
                with correct labels.
            y_pred: Tensor of shape '[batch_size, num_boxes, 4 + num_classes]'
                with predicted inferences.

        # Returns
            Tensor with loss per sample in batch.
        """
        regression_rotation = y_pred[:, :, :3]
        regression_translation = y_pred[:, :, 3:] 
        regression_translation = self.regress_translation(regression_translation, self.translation_priors)
        regression_translation = tf.convert_to_tensor(regression_translation)
        camera_parameter = self.compute_camera_parameter(y_true[0, 0, -1], LINEMOD_CAMERA_MATRIX, 1000)
        regression_translation = self.compute_tx_ty(regression_translation, camera_parameter)
        regression_translation = tf.convert_to_tensor(regression_translation)
        regression_translation = tf.expand_dims(regression_translation, axis = 0)

        regression_target_rotation = y_true[:, :, :3]
        regression_target_translation = y_true[:, :, 6:-2]
        is_symmetric = y_true[:, :, 3]
        class_indices = y_true[:, :, 4]
        anchor_state = tf.cast(tf.math.round(y_true[:, :, -2]), tf.int32)
        indices = tf.where(tf.equal(anchor_state, 1))
        regression_rotation = tf.gather_nd(regression_rotation, indices) * math.pi
        regression_translation = tf.gather_nd(regression_translation, indices)

        regression_target_rotation = tf.gather_nd(regression_target_rotation, indices) * math.pi
        regression_target_translation = tf.gather_nd(regression_target_translation, indices)
        is_symmetric = tf.gather_nd(is_symmetric, indices)
        is_symmetric = tf.cast(tf.math.round(is_symmetric), tf.int32)
        class_indices = tf.gather_nd(class_indices, indices)
        class_indices = tf.cast(tf.math.round(class_indices), tf.int32)
        axis_pred, angle_pred = self.separate_axis_from_angle(regression_rotation)
        axis_target, angle_target = self.separate_axis_from_angle(regression_target_rotation)
        selected_model_points = tf.gather(self.model_3d_points_for_loss, class_indices, axis = 0)
        axis_pred = tf.expand_dims(axis_pred, axis = 1)
        angle_pred = tf.expand_dims(angle_pred, axis = 1)
        axis_target = tf.expand_dims(axis_target, axis = 1)
        angle_target = tf.expand_dims(angle_target, axis = 1)

        regression_translation = tf.expand_dims(regression_translation, axis = 1)
        regression_target_translation = tf.expand_dims(regression_target_translation, axis = 1)

        transformed_points_pred = self.rotate(selected_model_points, axis_pred, angle_pred) + regression_translation
        transformed_points_target = self.rotate(selected_model_points, axis_target, angle_target) + regression_target_translation

        #distinct between symmetric and asymmetric objects
        sym_indices = tf.where(keras.backend.equal(is_symmetric, 1))
        asym_indices = tf.where(keras.backend.not_equal(is_symmetric, 1))

        num_points = selected_model_points.shape[1]
        sym_points_pred = tf.reshape(tf.gather_nd(transformed_points_pred, sym_indices), (-1, num_points, 3))
        asym_points_pred = tf.reshape(tf.gather_nd(transformed_points_pred, asym_indices), (-1, num_points, 3))

        sym_points_target = tf.reshape(tf.gather_nd(transformed_points_target, sym_indices), (-1, num_points, 3))
        asym_points_target = tf.reshape(tf.gather_nd(transformed_points_target, asym_indices), (-1, num_points, 3))

        sym_distances = self.calc_sym_distances(sym_points_pred, sym_points_target)
        asym_distances = self.calc_asym_distances(asym_points_pred, asym_points_target)

        # if (sym_distances is None) and (asym_distances is not None):
        #     to_concatenate = [asym_distances]

        # if (sym_distances is not None) and (asym_distances is None):
        #     to_concatenate = [sym_distances]

        # if (sym_distances is not None) and (asym_distances is not None):
        #     to_concatenate = [sym_distances, asym_distances]

        distances = tf.concat(asym_distances, axis = 0)
        loss = tf.math.reduce_mean(distances)
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        return loss

    def compute_tx_ty(self, translation_xy_Tz, camera_parameter):
        fx, fy = camera_parameter[0], camera_parameter[1],
        px, py = camera_parameter[2], camera_parameter[3],
        tz_scale, image_scale = camera_parameter[4], camera_parameter[5]

        x = translation_xy_Tz[:, 0] / tf.cast(image_scale, tf.float32)
        y = translation_xy_Tz[:, 1] / tf.cast(image_scale, tf.float32)
        tz = translation_xy_Tz[:, 2] * tz_scale

        x = x - px
        y = y - py

        tx = np.multiply(x, tz) / fx
        ty = np.multiply(y, tz) / fy

        tx = tf.expand_dims(tx, axis=0)
        ty = tf.expand_dims(ty, axis=0)
        tz = tf.expand_dims(tz, axis=0)

        translations = tf.concat([tx, ty, tz], axis=0)
        return tf.transpose(translations)

    def regress_translation(self, translation_raw, translation_priors):
        stride = translation_priors[:, -1]
        x = translation_priors[:, 0] + (translation_raw[:, :, 0] * stride)
        y = translation_priors[:, 1] + (translation_raw[:, :, 1] * stride)
        Tz = translation_raw[:, :, 2]
        translations_predicted = tf.concat([x, y, Tz], axis=0)
        return tf.transpose(translations_predicted)

    def compute_camera_parameter(self, image_scale, camera_matrix,
                                 translation_scale_norm):
        camera_parameter = tf.convert_to_tensor([camera_matrix[0, 0],
                                                 camera_matrix[1, 1],
                                                 camera_matrix[0, 2],
                                                 camera_matrix[1, 2],
                                                 translation_scale_norm,
                                                 image_scale])
        return camera_parameter

    def separate_axis_from_angle(self, axis_angle_tensor):
        squared = tf.math.square(axis_angle_tensor)
        summed = tf.math.reduce_sum(squared, axis = -1)
        angle = tf.expand_dims(tf.math.sqrt(summed), axis = -1)
        axis = tf.math.divide_no_nan(axis_angle_tensor, angle)
        return axis, angle

    def rotate(self, point, axis, angle, name=None):
        with tf.compat.v1.name_scope(name, "axis_angle_rotate", [point, axis, angle]):
            cos_angle = tf.cos(angle)
            axis_dot_point = self.dot(axis, point)
            return point * cos_angle + self.cross(
                axis, point) * tf.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)

    def dot(self, vector1, vector2, axis=-1, keepdims=True, name=None):
        with tf.compat.v1.name_scope(name, "vector_dot", [vector1, vector2]):
            return tf.reduce_sum(input_tensor=vector1 * vector2, axis=axis, keepdims=keepdims)

    def cross(self, vector1, vector2, name=None):
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
            return tf.stack((n_x, n_y, n_z), axis = -1)

    def calc_sym_distances(self, sym_points_pred, sym_points_target):
        sym_points_pred = tf.expand_dims(sym_points_pred, axis = 2)
        sym_points_target = tf.expand_dims(sym_points_target, axis = 1)
        distances = tf.reduce_min(tf.norm(sym_points_pred - sym_points_target, axis = -1), axis = -1)

    def calc_asym_distances(self, asym_points_pred, asym_points_target):
        distances = tf.norm(asym_points_pred - asym_points_target, axis = -1)
        return tf.reduce_mean(distances, axis = -1)