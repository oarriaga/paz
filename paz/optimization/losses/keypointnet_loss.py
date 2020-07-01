import math
import tensorflow as tf
import numpy as np

from ...models.keypoint.projector import Projector


class KeypointNetLoss(object):
    """KeypointNet loss for discovering latent keypoints.

    # Arguments
        num_keypints: Int. Number of keypoints to discover.
        focal_length: Float. Focal length of camera
        rotation_noise: Float. Noise added to the estimation of the rotation.
        separation_delta: Float. Delta used for the ''separation'' loss.
        loss_weights: Dict. having as keys strings with the different losses
            names e.g. ''consistency'' and as value the weight used for that
            loss.

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/pdf/1807.03146.pdf)
    """
    def __init__(self, num_keypoints, focal_length, rotation_noise=0.1,
                 separation_delta=0.05, loss_weights={
                     'consistency': 1.0, 'silhouette': 1.0, 'separation': 1.0,
                     'relative_pose': 0.2, 'variance': 0.5}):

        self.num_keypoints = int(num_keypoints)
        self.focal_length = focal_length
        self.projector = Projector(focal_length)
        self.rotation_noise = rotation_noise
        self.separation_delta = separation_delta
        self.loss_weights = loss_weights

    def _reshape_matrix(self, matrix):
        matrix = tf.reshape(matrix, [-1, 4, 4])
        # transpose is for multiplying points with matrices from the left.
        matrix = tf.transpose(matrix, [0, 2, 1])
        return matrix

    def _unpack_matrices(self, matrices):
        world_to_A = self._reshape_matrix(matrices[:, 0, :])
        world_to_B = self._reshape_matrix(matrices[:, 1, :])
        A_to_world = self._reshape_matrix(matrices[:, 2, :])
        B_to_world = self._reshape_matrix(matrices[:, 3, :])
        return world_to_A, world_to_B, A_to_world, B_to_world

    def _unpack_uvz_coordinates(self, uvz_coordinates):
        uvz_A = uvz_coordinates[:, :self.num_keypoints, :]
        uvz_B = uvz_coordinates[:, self.num_keypoints:, :]
        return uvz_A, uvz_B

    def _consistency(self, uvz_M, M_to_world, world_to_N, uvz_N):
        keypoints_M = self.projector.unproject(uvz_M)
        world_coordinates = tf.matmul(keypoints_M, M_to_world)
        keypoints_M_in_N = tf.matmul(world_coordinates, world_to_N)
        uvz_M_in_N = self.projector.project(keypoints_M_in_N)
        squared_difference = tf.square(uvz_M_in_N - uvz_N)
        l2_distance = tf.reduce_sum(squared_difference, axis=[1, 2])
        consistency_loss = l2_distance / self.num_keypoints
        return consistency_loss

    def consistency(self, matrices, uvz_coordinates):
        matrices = self._unpack_matrices(matrices)
        world_to_A, world_to_B, A_to_world, B_to_world = matrices
        uvz_A, uvz_B = self._unpack_uvz_coordinates(uvz_coordinates)
        consistency_A = self._consistency(uvz_A, A_to_world, world_to_B, uvz_B)
        consistency_B = self._consistency(uvz_B, B_to_world, world_to_A, uvz_A)
        consistency_loss = (consistency_A + consistency_B) / 2.0
        consistency_loss = self.loss_weights['consistency'] * consistency_loss
        return consistency_loss

    def _separation(self, uvz):
        keypoints = self.projector.unproject(uvz)
        keypoints_i = tf.tile(keypoints, [1, self.num_keypoints, 1])
        keypoints_j = tf.tile(keypoints, [1, 1, self.num_keypoints])
        keypoints_j = tf.reshape(keypoints_j, tf.shape(keypoints_i))
        squared_difference = tf.square(keypoints_i - keypoints_j)
        squared_l2_distance = tf.reduce_sum(squared_difference, axis=2)
        separation_loss = tf.maximum(
            -squared_l2_distance + self.separation_delta, 0.0)
        separation_loss = tf.reshape(
            separation_loss, [-1, self.num_keypoints, self.num_keypoints])
        separation_loss = tf.reduce_sum(separation_loss, axis=[1, 2])
        separation_loss = separation_loss / self.num_keypoints
        return separation_loss

    def separation(self, matrices, uvz_coordinates):
        uvz_A, uvz_B = self._unpack_uvz_coordinates(uvz_coordinates)
        separation_loss_A = self._separation(uvz_A)
        separation_loss_B = self._separation(uvz_B)
        separation_loss = (separation_loss_A + separation_loss_B) / 2.0
        separation_loss = self.loss_weights['separation'] * separation_loss
        return separation_loss

    def relative_pose(self, matrices, uvz_coordinates):
        matrices = self._unpack_matrices(matrices)
        world_to_A, world_to_B, A_to_world, B_to_world = matrices
        uvz_A, uvz_B = self._unpack_uvz_coordinates(uvz_coordinates)
        keypoints_A = self.projector.unproject(uvz_A)
        keypoints_B = self.projector.unproject(uvz_B)

        A_to_B = tf.matmul(A_to_world, world_to_B)
        rotation_A_to_B = A_to_B[:, :3, :3]
        estimation_args = (keypoints_A, keypoints_B, self.rotation_noise)
        estimated_rotation_A_to_B = self.estimate_rotation(*estimation_args)
        estimated_rotation_A_to_B = estimated_rotation_A_to_B[:, :3, :3]
        squared_A_to_B = tf.square(estimated_rotation_A_to_B - rotation_A_to_B)
        squared_frobenius = tf.reduce_sum(squared_A_to_B, axis=[1, 2])
        frobenius = tf.sqrt(squared_frobenius)
        arcsin_arg = tf.minimum(1.0, frobenius / (2 * math.sqrt(2)))
        angular_loss = 2.0 * tf.asin(arcsin_arg)
        angular_loss = self.loss_weights['relative_pose'] * angular_loss
        return angular_loss

    def uvz_points(self, matrices, uvz_coordinates):
        consistency_loss = self.consistency(matrices, uvz_coordinates)
        separation_loss = self.separation(matrices, uvz_coordinates)
        relative_pose_loss = self.relative_pose(matrices, uvz_coordinates)
        uvz_loss = consistency_loss + separation_loss + relative_pose_loss
        return uvz_loss

    def _silhouette(self, alpha_channel, uv_volume):
        alpha_channel = tf.greater(alpha_channel, tf.zeros_like(alpha_channel))
        alpha_channel = tf.cast(alpha_channel, dtype=tf.float32)
        alpha_channel = tf.expand_dims(alpha_channel, 1)
        silhouette_loss = tf.reduce_sum(uv_volume * alpha_channel, axis=[2, 3])
        silhouette_loss = -tf.math.log(silhouette_loss + 1e-12)
        silhouette_loss = tf.reduce_mean(silhouette_loss, axis=-1)
        return silhouette_loss

    def silhouette(self, alpha_channels, uv_volumes):
        alpha_channel_A = alpha_channels[:, :, :, 0]
        alpha_channel_B = alpha_channels[:, :, :, 1]
        uv_volume_A = uv_volumes[:, :self.num_keypoints, :, :]
        uv_volume_B = uv_volumes[:, self.num_keypoints:, :, :]
        silhouette_loss_A = self._silhouette(alpha_channel_A, uv_volume_A)
        silhouette_loss_B = self._silhouette(alpha_channel_B, uv_volume_B)
        silhouette_loss = (silhouette_loss_A + silhouette_loss_B) / 2.0
        silhouette_loss = self.loss_weights['silhouette'] * silhouette_loss
        return silhouette_loss

    def _variance(self, uv_volume, range_u, range_v):
        expected_keypoint_u = tf.reduce_sum(uv_volume * range_u, axis=[2, 3])
        expected_keypoint_v = tf.reduce_sum(uv_volume * range_v, axis=[2, 3])
        uv = tf.stack([expected_keypoint_u, expected_keypoint_v], -1)
        uv = tf.reshape(uv, [-1, self.num_keypoints, 2])
        uv = tf.reshape(uv, [tf.shape(uv)[0], tf.shape(uv)[1], 1, 1, 2])

        ranges = tf.stack([range_u, range_v], axis=2)
        ranges_sh = tf.shape(ranges)
        ranges = tf.reshape(ranges, [1, 1, ranges_sh[0], ranges_sh[1], 2])
        squared_difference = tf.reduce_sum(tf.square(uv - ranges), axis=4)
        diff = squared_difference * uv_volume
        diff = tf.reduce_sum(diff, axis=[2, 3])
        variance = tf.reduce_mean(diff, axis=-1)
        return variance

    def variance(self, alpha_channels, uv_volumes):
        uv_volume_A = uv_volumes[:, :self.num_keypoints, :, :]
        uv_volume_B = uv_volumes[:, self.num_keypoints:, :, :]
        feature_map_size = uv_volumes.shape[-1]

        arange = np.arange(0.5, feature_map_size, 1)
        arange = arange / (feature_map_size / 2) - 1
        range_u, range_v = tf.meshgrid(arange, -arange)
        range_u = tf.cast(range_u, dtype=tf.float32)
        range_v = tf.cast(range_v, dtype=tf.float32)

        variance_loss_A = self._variance(uv_volume_A, range_u, range_v)
        variance_loss_B = self._variance(uv_volume_B, range_u, range_v)
        variance_loss = (variance_loss_A + variance_loss_B) / 2.0
        variance_loss = self.loss_weights['variance'] * variance_loss
        return variance_loss

    def uv_volumes(self, alpha_channels, uv_volumes):
        variance_loss = self.variance(alpha_channels, uv_volumes)
        silhouette_loss = self.silhouette(alpha_channels, uv_volumes)
        uv_loss = variance_loss + silhouette_loss
        return uv_loss

    def estimate_rotation(self, keypoints_A, keypoints_B, noise=0.1):
        """Estimates the rotation between two sets of keypoints using
        Kabsch algorithm.

        The rotation is estimated by first subtracting mean from each
        set of keypoints and computing SVD of the covariance matrix.

        Arguments:
            xyz0: [batch, num_kp, 3] The first set of keypoints.
            xyz1: [batch, num_kp, 3] The second set of keypoints.
            pconf: [batch, num_kp] The weights used to
                   compute the rotation estimate.
            noise: A number indicating the noise added to the keypoints.

        Returns:
            [batch, 3, 3] A batch of transposed 3 x 3 rotation matrices.
        """

        pconf = tf.ones(
            [tf.shape(keypoints_A)[0],
             tf.shape(keypoints_A)[1]],
            dtype=tf.float32) / self.num_keypoints

        noise_A = tf.random.normal(tf.shape(keypoints_A), mean=0, stddev=noise)
        noise_B = tf.random.normal(tf.shape(keypoints_B), mean=0, stddev=noise)
        keypoints_A = keypoints_A + noise_A
        keypoints_B = keypoints_B + noise_B
        pconf2 = tf.expand_dims(pconf, 2)
        center_A = tf.reduce_sum(pconf2 * keypoints_A, 1, keepdims=True)
        center_B = tf.reduce_sum(pconf2 * keypoints_B, 1, keepdims=True)
        x = keypoints_A - center_A
        y = keypoints_B - center_B
        weighted_x = tf.matmul(
            x, tf.linalg.diag(pconf), transpose_a=True)
        covariance = tf.matmul(weighted_x, y)
        _, u, v = tf.linalg.svd(covariance, full_matrices=True)
        d = tf.linalg.det(tf.matmul(v, u, transpose_b=True))
        ud = tf.concat(
            [u[:, :, :-1],
             u[:, :, -1:] * tf.expand_dims(tf.expand_dims(d, 1), 1)],
            axis=2)
        return tf.matmul(ud, v, transpose_b=True)
