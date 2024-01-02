import tensorflow as tf
import numpy as np
from plyfile import PlyData
from pose import LINEMOD_CAMERA_MATRIX


class MultiPoseLoss(object):
    def __init__(self, object_id, translation_priors, data_path,
                 target_num_points=500, num_pose_dims=3, model_path='models/',
                 translation_scale_norm=1000):
        self.object_id = object_id
        self.translation_priors = translation_priors
        self.num_pose_dims = num_pose_dims
        self.tz_scale = tf.convert_to_tensor(translation_scale_norm,
                                             dtype=tf.float32)
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

    def _compute_translation(self, translation_raw_pred, scale):
        camera_matrix = tf.convert_to_tensor(LINEMOD_CAMERA_MATRIX)
        translation_pred = self._regress_translation(translation_raw_pred)
        translation_pred = self._compute_tx_ty(translation_pred, camera_matrix,
                                               scale)
        return translation_pred

    def _regress_translation(self, translation_raw):
        stride = self.translation_priors[:, -1]
        x = self.translation_priors[:, 0] + (translation_raw[:, :, 0] * stride)
        y = self.translation_priors[:, 1] + (translation_raw[:, :, 1] * stride)
        x, y = x[:, :, tf.newaxis], y[:, :, tf.newaxis]
        Tz = translation_raw[:, :, 2]
        Tz = Tz[:, :, tf.newaxis]
        translations_predicted = tf.concat([x, y, Tz], axis=-1)
        return translations_predicted

    def _compute_tx_ty(self, translation_xy_Tz, camera_matrix, scale):
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        px, py = camera_matrix[0, 2], camera_matrix[1, 2]

        x = translation_xy_Tz[:, :, 0] / scale
        y = translation_xy_Tz[:, :, 1] / scale
        tz = translation_xy_Tz[:, :, 2] * self.tz_scale

        x = x - px
        y = y - py

        tx = tf.math.multiply(x, tz) / fx
        ty = tf.math.multiply(y, tz) / fy

        tx = tx[:, :, tf.newaxis]
        ty = ty[:, :, tf.newaxis]
        tz = tz[:, :, tf.newaxis]
        translations = tf.concat([tx, ty, tz], axis=-1)
        return translations

    def _separate_axis_from_angle(self, axis_angle):
        squared = tf.math.square(axis_angle)
        sum = tf.math.reduce_sum(squared, axis=-1)
        angle = tf.expand_dims(tf.math.sqrt(sum), axis=-1)
        axis = tf.math.divide_no_nan(axis_angle, angle)
        return axis, angle

    def _rotate(self, point, axis, angle):
        cos_angle = tf.cos(angle)
        axis_dot_point = self._dot(axis, point)
        points_rotated = (
            point * cos_angle + self._cross(axis, point) * tf.sin(angle) +
            axis * axis_dot_point * (1.0 - cos_angle))
        return points_rotated

    def _dot(self, vector1, vector2, axis=-1, keepdims=True):
        return tf.reduce_sum(input_tensor=vector1 * vector2,
                             axis=axis, keepdims=keepdims)

    def _cross(self, vector1, vector2):
        vector1_x, vector1_y,  = vector1[:, :, 0], vector1[:, :, 1]
        vector1_z = vector1[:, :, 2]
        vector2_x, vector2_y = vector2[:, :, 0], vector2[:, :, 1]
        vector2_z = vector2[:, :, 2]
        n_x = vector1_y * vector2_z - vector1_z * vector2_y
        n_y = vector1_z * vector2_x - vector1_x * vector2_z
        n_z = vector1_x * vector2_y - vector1_y * vector2_x
        return tf.stack((n_x, n_y, n_z), axis=-1)

    def _calc_sym_distances(self, sym_points_pred, sym_points_true):
        sym_points_pred = sym_points_pred[:, :, tf.newaxis]
        sym_points_true = sym_points_true[:, tf.newaxis]
        distances = tf.reduce_min(tf.norm(sym_points_pred - sym_points_true,
                                          axis=-1), axis=-1)
        return tf.reduce_mean(distances, axis=-1)

    def _calc_asym_distances(self, asym_points_pred, asym_points_target):
        distances = tf.norm(asym_points_pred - asym_points_target, axis=-1)
        return tf.reduce_mean(distances, axis=-1)

    def compute_loss(self, y_true, y_pred):
        rotation_pred = y_pred[:, :, :self.num_pose_dims]
        rotation_true = y_true[:, :, :self.num_pose_dims]
        translation_true = y_true[:, :, 2 * self.num_pose_dims:2 *
                                  self.num_pose_dims + self.num_pose_dims]
        translation_raw_pred = y_pred[:, :, self.num_pose_dims:]
        scale = y_true[0, 0, -1]
        translation_pred = self._compute_translation(translation_raw_pred,
                                                     scale)

        anchor_flags = y_true[:, :, -2]
        anchor_state = tf.cast(tf.math.round(anchor_flags), tf.int32)
        indices = tf.where(tf.equal(anchor_state, 1))

        rotation_pred = tf.gather_nd(rotation_pred, indices)
        rotation_pred = rotation_pred * np.pi
        rotation_true = tf.gather_nd(rotation_true, indices)
        rotation_true = rotation_true * np.pi
        translation_pred = tf.gather_nd(translation_pred, indices)
        translation_true = tf.gather_nd(translation_true, indices)

        is_symmetric = y_true[:, :, self.num_pose_dims]
        is_symmetric = tf.gather_nd(is_symmetric, indices)
        is_symmetric = tf.cast(tf.math.round(is_symmetric), tf.int32)
        class_indices = y_true[:, :, self.num_pose_dims + 1]
        class_indices = tf.gather_nd(class_indices, indices)
        class_indices = tf.cast(tf.math.round(class_indices), tf.int32)

        axis_pred, angle_pred = self._separate_axis_from_angle(rotation_pred)
        axis_true, angle_true = self._separate_axis_from_angle(rotation_true)

        axis_pred = axis_pred[:, tf.newaxis, :]
        axis_true = axis_true[:, tf.newaxis, :]
        angle_pred = angle_pred[:, tf.newaxis, :]
        angle_true = angle_true[:, tf.newaxis, :]

        translation_pred = translation_pred[:, tf.newaxis, :]
        translation_true = translation_true[:, tf.newaxis, :]

        selected_model_points = tf.gather(self.model_points,
                                          class_indices, axis=0)
        transformed_points_pred = self._rotate(
            selected_model_points, axis_pred, angle_pred) + translation_pred
        transformed_points_true = (self._rotate(
            selected_model_points, axis_true, angle_true) + translation_true)

        num_points = selected_model_points.shape[1]
        sym_indices = tf.where(tf.math.equal(is_symmetric, 1))
        sym_points_pred = tf.reshape(tf.gather_nd(
            transformed_points_pred, sym_indices), (-1, num_points, 3))
        sym_points_true = tf.reshape(tf.gather_nd(
            transformed_points_true, sym_indices), (-1, num_points, 3))

        asym_indices = tf.where(tf.math.not_equal(is_symmetric, 1))
        asym_points_pred = tf.reshape(tf.gather_nd(
            transformed_points_pred, asym_indices), (-1, num_points, 3))
        asym_points_true = tf.reshape(tf.gather_nd(
            transformed_points_true, asym_indices), (-1, num_points, 3))

        sym_distances = self._calc_sym_distances(
            sym_points_pred, sym_points_true)
        asym_distances = self._calc_asym_distances(
            asym_points_pred, asym_points_true)

        distances = tf.concat([sym_distances, asym_distances], axis=0)
        loss = tf.math.reduce_mean(distances)
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        return loss


if __name__ == "__main__":
    import pickle
    from pose import EFFICIENTPOSEA

    with open('y_pred.pkl', 'rb') as f:
        y_pred = pickle.load(f)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    with open('y_true.pkl', 'rb') as f:
        y_true = pickle.load(f)
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    model = EFFICIENTPOSEA(2, base_weights='COCO', head_weights=None)
    pose_loss = MultiPoseLoss('08', model.translation_priors,
                              'Linemod_preprocessed/')
    loss = pose_loss.compute_loss(y_true, y_pred)         # should be 1038.3807
    print('nj')
