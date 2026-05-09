import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Huber
from examples.frustum_pointnet.dataset_utils import NUM_HEADING_BIN, \
    g_mean_size_arr, NUM_SIZE_CLUSTER

def ExtractBox3DCornersHelper(centers, headings, sizes):
    """ TF layer.
        Inputs:
            center: (B,3)
            heading_residuals: (B,NH)
            size_residuals: (B,NS,3)
        Outputs:
            box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = centers.get_shape()[0]
    length = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    width = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    height = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    x_corners = tf.concat([length / 2, length / 2, -length / 2, -length / 2,
                           length / 2, length / 2, -length / 2, -length / 2],
                          axis=1)  # (N,8)
    y_corners = tf.concat([height / 2, height / 2, height / 2, height / 2,
                           -height / 2, -height / 2, -height / 2, -height / 2],
                          axis=1)  # (N,8)
    z_corners = tf.concat([width / 2, -width / 2, -width / 2, width / 2,
                           width / 2, -width / 2, -width / 2, width / 2],
                          axis=1)  # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners, 1),
                         tf.expand_dims(y_corners, 1),
                         tf.expand_dims(z_corners, 1)],
                        axis=1)  # (N,3,8)
    cosine_value = tf.cos(headings)
    sine_value = tf.sin(headings)
    ones = tf.ones([batch_size], dtype=tf.float32)
    zeros = tf.zeros([batch_size], dtype=tf.float32)
    row1 = tf.stack([cosine_value, zeros, sine_value], axis=1)  # (N,3)
    row2 = tf.stack([zeros, ones, zeros], axis=1)
    row3 = tf.stack([-sine_value, zeros, cosine_value], axis=1)
    R = tf.concat([tf.expand_dims(row1, 1), tf.expand_dims(row2, 1),
                   tf.expand_dims(row3, 1)], axis=1)  # (N,3,3)
    corners_3d = tf.matmul(R, corners)  # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers, 2), [1, 1, 8])  # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0, 2, 1])  # (N,8,3)
    return corners_3d


def ExtractBox3DCorners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0]
    heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi /
                                                NUM_HEADING_BIN),
                                      dtype=tf.float32)  # (NH,)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0)

    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32),
                                0) + size_residuals  # (B,NS,1)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes, 1), [1, NUM_HEADING_BIN, 1, 1])
    headings = tf.tile(tf.expand_dims(headings, -1), [1, 1, NUM_SIZE_CLUSTER])
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center, 1), 1),
                      [1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1])

    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = ExtractBox3DCornersHelper(tf.reshape(centers, [N, 3]),
                                           tf.reshape(headings, [N]),
                                           tf.reshape(sizes, [N, 3]))

    return tf.reshape(corners_3d,
                      [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])


class FrustumPointNetLoss:
    """ Loss functions for 3D object detection.
        Input:
            args: dict
                    mask_label: TF int32 tensor in shape (B,N)
                    center_label: TF tensor in shape (B,3)
                    heading_class_label: TF int32 tensor in shape (B,)
                    heading_residual_label: TF tensor in shape (B,)
                    size_class_label: TF tensor int32 in shape (B,)
                    size_residual_label: TF tensor tensor in shape (B,)
                    end_points: dict, outputs from our model
            corner_loss_weight: float scalar
            box_loss_weight: float scalar

        Output:
            total_loss: TF scalar tensor
                the total_loss is also added to the losses collection

        References
            - [Frustum PointNets for 3D Object Detection from RGB-D Data]
            (https://arxiv.org/abs/1711.08488)
    """

    def __init__(self, corner_loss_weight=10.0, box_loss_weight=1.0,
                 mask_weight=1.0):
        self.corner_loss_weight = corner_loss_weight
        self.box_loss_weight = box_loss_weight
        self.mask_weight = mask_weight

    def _huber_loss(self, y_true, y_pred):
        """
        Huber Loss calculation for regressed output
        @param y_true: Label of the loss
        @param y_pred: Prediction from Frustum-PointNet
        @return: Loss value
        """
        h = Huber()
        return h(y_true, y_pred)

    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Cross Entropy Loss calculation for classification output
        @param y_true: Label of the loss
        @param y_pred: Prediction from Frustum-PointNet
        @return: Loss value
        """
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                              (logits=y_pred, labels=tf.cast(y_true, tf.int64)))
        return loss

    def to_one_hot(self, input_tensor, depth):
        OneHot = tf.one_hot(tf.cast(input_tensor, tf.int64), depth=depth,
                            on_value=1, off_value=0, axis=-1)
        return OneHot

    def TotalLoss(self, args):
        mask_label, center_label, heading_class_label, heading_residual_label, \
        size_class_label, size_residual_label, end_points = args[0], args[1], \
                                                            args[2], args[3], \
                                                            args[4], args[5], \
                                                            args[6]

        mask_loss = self._cross_entropy_loss(y_true=tf.cast(mask_label,
                                                            tf.int64),
                                             y_pred=end_points['mask_logits'])

        center_loss = self._huber_loss(y_true=center_label,
                                       y_pred=end_points['center'])

        stage1_center_loss = self._huber_loss(y_true=center_label,
                                              y_pred=end_points[
                                                  'stage1_center'])

        heading_class_loss = self._cross_entropy_loss(
            y_true=tf.cast(heading_class_label, tf.int64),
            y_pred=end_points['heading_scores'])

        heading_cls_onehot = self.to_one_hot(heading_class_label,
                                             NUM_HEADING_BIN)

        heading_residual_normalized_label = (heading_residual_label /
                                             (np.pi / NUM_HEADING_BIN))

        heading_residual_normalized_loss = self._huber_loss(
            y_true=heading_residual_normalized_label, y_pred=tf.reduce_sum(
                end_points['heading_residuals_normalized'] *
                tf.cast(heading_cls_onehot, dtype=tf.float32), axis=1))

        size_class_loss = self._cross_entropy_loss(
            y_pred=end_points['size_scores'], y_true=tf.cast(size_class_label,
                                                             tf.int64))

        size_cls_onehot = self.to_one_hot(tf.cast(size_class_label, tf.int64),
                                          NUM_SIZE_CLUSTER)

        size_cls_onehot_tiled = tf.tile(tf.expand_dims(tf.cast(size_cls_onehot,
                                                               dtype=tf.float32)
                                                       ,-1), [1, 1, 3])
        predicted_size_residual_normalized = tf.reduce_sum(
            end_points['size_residuals_normalized'] * size_cls_onehot_tiled,
            axis=[1])

        mean_size_arr_expand = tf.expand_dims(tf.constant(g_mean_size_arr,
                                                          dtype=tf.float32), 0)

        mean_size_label = tf.reduce_sum(size_cls_onehot_tiled *
                                        mean_size_arr_expand, axis=[1])

        size_residual_label_normalized = size_residual_label / mean_size_label

        size_residual_normalized_loss = self._huber_loss(
            y_true=size_residual_label_normalized,
            y_pred=predicted_size_residual_normalized)

        corners_3d = ExtractBox3DCorners(end_points['center'],
                                         end_points['heading_residuals'],
                                         end_points['size_residuals'])

        gt_mask = tf.tile(tf.expand_dims(heading_cls_onehot, 2),
                          [1, 1, NUM_SIZE_CLUSTER]) * tf.tile(tf.expand_dims(
            size_cls_onehot, 1), [1, NUM_HEADING_BIN, 1])

        corners_3d_pred = tf.reduce_sum(tf.cast(tf.expand_dims(
            tf.expand_dims(gt_mask, -1), -1), dtype=tf.float32) * corners_3d,
                                        axis=[1, 2])

        heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi /
                                                    NUM_HEADING_BIN),
                                          dtype=tf.float32)  # (NH,)

        heading_label = tf.expand_dims(heading_residual_label, 1) + \
                        tf.expand_dims(heading_bin_centers, 0)  # (B,NH)

        heading_label = tf.reduce_sum(tf.cast(heading_cls_onehot,
                                              dtype=tf.float32) * heading_label,
                                      1)

        mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr,
                                                dtype=tf.float32),0)  # (1,NS,3)
        size_label = mean_sizes + tf.expand_dims(size_residual_label, 1)
        size_label = tf.reduce_sum(tf.expand_dims(tf.cast(size_cls_onehot,
                                                          dtype=tf.float32),
                                                  -1) * size_label, axis=[1])

        corners_3d_gt = ExtractBox3DCornersHelper(center_label, heading_label,
                                                  size_label)  # (B,8,3)
        corners_3d_gt_flip = ExtractBox3DCornersHelper(center_label,
                                                       heading_label
                                                       + np.pi, size_label)

        corners_3D_loss = self._huber_loss(y_true=corners_3d_gt,
                                           y_pred=corners_3d_pred)
        corners_3D_flip_loss = self._huber_loss(y_true=corners_3d_gt_flip,
                                                y_pred=corners_3d_pred)

        corners_loss = tf.minimum(corners_3D_loss, corners_3D_flip_loss)

        total_loss = mask_loss * self.mask_weight + self.box_loss_weight * \
                     (center_loss + heading_class_loss + size_class_loss +
                      heading_residual_normalized_loss * 20 +
                      size_residual_normalized_loss * 20 + stage1_center_loss +
                      self.corner_loss_weight * corners_loss)

        return total_loss
