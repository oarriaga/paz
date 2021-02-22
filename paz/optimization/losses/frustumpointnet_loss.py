import numpy as np
import tensorflow as tf

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8
NUM_OBJECT_POINT = 512

type2class = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3,
              'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7}
class2type = {type2class[t]: t for t in type2class}
type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191]),
                  'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                  'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                  'Pedestrian': np.array([0.84422524, 0.66068622, 1.7625519]),
                  'Person_sitting': np.array([0.80057803, 0.5983815, 1.2745]),
                  'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                  'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                  'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = type_mean_size[class2type[i]]


def ExtractBox3DCornersHelper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    batch_size = centers.get_shape()[0]
    length = tf.slice(sizes, [0, 0], [-1, 1])  # (N,1)
    width = tf.slice(sizes, [0, 1], [-1, 1])  # (N,1)
    height = tf.slice(sizes, [0, 2], [-1, 1])  # (N,1)
    # print l,w,h
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

    def HuberLoss(self, error, delta):
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        return tf.reduce_mean(losses)

    def CrossEntropyLoss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                              (logits=logits, labels=tf.cast(labels, tf.int64)))
        return loss

    def ToOneHot(self, input_tensor, depth):
        OneHot = tf.one_hot(tf.cast(input_tensor, tf.int64), depth=depth,
                            on_value=1, off_value=0, axis=-1)
        return OneHot

    def TotalLoss(self, args):
        mask_label, center_label, heading_class_label, heading_residual_label, \
        size_class_label, size_residual_label, end_points = args[0], args[1], \
                                                            args[2], args[3], \
                                                            args[4], args[5], \
                                                            args[6]

        mask_loss = self.CrossEntropyLoss(logits=end_points['mask_logits'],
                                          labels=tf.cast(mask_label, tf.int64))

        center_dist = tf.norm(center_label - end_points['center'], axis=-1)
        center_loss = self.HuberLoss(center_dist, delta=2.0)

        stage1_center_dist = tf.norm(center_label - end_points['stage1_center'],
                                     axis=-1)
        stage1_center_loss = self.HuberLoss(stage1_center_dist, delta=1.0)

        heading_class_loss = self.CrossEntropyLoss(
            logits=end_points['heading_scores'], labels=tf.cast(
                heading_class_label, tf.int64))

        hcls_onehot = self.ToOneHot(heading_class_label, NUM_HEADING_BIN)

        heading_residual_normalized_label = (heading_residual_label /
                                             (np.pi / NUM_HEADING_BIN))

        heading_residual_normalized_loss = self.HuberLoss(error=tf.reduce_sum(
            end_points['heading_residuals_normalized'] *
            tf.cast(hcls_onehot, dtype=tf.float32),
            axis=1) - heading_residual_normalized_label, delta=1.0)

        size_class_loss = self.CrossEntropyLoss(
            logits=end_points['size_scores'], labels=tf.cast(size_class_label,
                                                             tf.int64))

        scls_onehot = self.ToOneHot(tf.cast(size_class_label, tf.int64),
                                    NUM_SIZE_CLUSTER)

        scls_onehot_tiled = tf.tile(tf.expand_dims(tf.cast(scls_onehot,
                                                           dtype=tf.float32),
                                                   -1), [1, 1, 3])
        predicted_size_residual_normalized = tf.reduce_sum(
            end_points['size_residuals_normalized'] * scls_onehot_tiled,
            axis=[1])

        mean_size_arr_expand = tf.expand_dims(tf.constant(g_mean_size_arr,
                                                          dtype=tf.float32), 0)

        mean_size_label = tf.reduce_sum(scls_onehot_tiled *
                                        mean_size_arr_expand, axis=[1])

        size_residual_label_normalized = size_residual_label / mean_size_label

        size_normalized_dist = tf.norm(size_residual_label_normalized -
                                       predicted_size_residual_normalized,
                                       axis=-1)

        size_residual_normalized_loss = self.HuberLoss(size_normalized_dist,
                                                       delta=1.0)

        corners_3d = ExtractBox3DCorners(end_points['center'],
                                         end_points['heading_residuals'],
                                         end_points['size_residuals'])

        gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2),
                          [1, 1, NUM_SIZE_CLUSTER]) * tf.tile(tf.expand_dims(
            scls_onehot, 1), [1, NUM_HEADING_BIN, 1])

        corners_3d_pred = tf.reduce_sum(tf.cast(tf.expand_dims(
            tf.expand_dims(gt_mask, -1), -1), dtype=tf.float32) * corners_3d,
                                        axis=[1, 2])

        heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi /
                                                    NUM_HEADING_BIN),
                                          dtype=tf.float32)  # (NH,)

        heading_label = tf.expand_dims(heading_residual_label, 1) + \
                        tf.expand_dims(heading_bin_centers, 0)  # (B,NH)

        heading_label = tf.reduce_sum(tf.cast(hcls_onehot, dtype=tf.float32) *
                                      heading_label, 1)

        mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr,
                                                dtype=tf.float32),
                                    0)  # (1,NS,3)
        size_label = mean_sizes + tf.expand_dims(size_residual_label, 1)
        size_label = tf.reduce_sum(tf.expand_dims(tf.cast(scls_onehot,
                                                          dtype=tf.float32),
                                                  -1) * size_label, axis=[1])

        corners_3d_gt = ExtractBox3DCornersHelper(center_label, heading_label,
                                                  size_label)  # (B,8,3)
        corners_3d_gt_flip = ExtractBox3DCornersHelper(center_label,
                                                       heading_label
                                                       + np.pi, size_label)

        corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt,
                                          axis=-1),
                                  tf.norm(corners_3d_pred - corners_3d_gt_flip,
                                          axis=-1))

        corners_loss = self.HuberLoss(corners_dist, delta=1.0)

        total_loss = mask_loss * self.mask_weight + self.box_loss_weight * \
                     (center_loss + heading_class_loss + size_class_loss +
                      heading_residual_normalized_loss * 20 +
                      size_residual_normalized_loss * 20 + stage1_center_loss +
                      self.corner_loss_weight * corners_loss)

        return

    class FrustumPointNetLoss(object):
        def __init__(self, loss_weights=None):
            if loss_weights is None:
                loss_weights = {'corner_loss_weight': 10.0,
                                'box_loss_weight': 1.0,
                                'mask_weight': 1.0}

        def _huber_loss(self, y_true, y_pred):
            huberloss = tf.keras.losses.Huber(reduction=
                                              tf.keras.losses.Reduction.SUM)
            return huberloss(y_true, y_pred).numpy()

        def _cross_entropy(self, y_true, y_pred):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                                  (logits=y_pred,
                                   labels=tf.cast(y_true, tf.int64)))
            return loss

        def _to_onehot(self, input_tensor, depth):
            OneHot = tf.one_hot(tf.cast(input_tensor, tf.int64), depth=depth,
                                on_value=1, off_value=0, axis=-1)
            return OneHot

        def mask_loss(self, y_true, y_pred):
            loss = self._cross_entropy(y_true, y_pred)
            return loss

        def center_loss(self, y_true, y_pred):
            loss = self._huber_loss(y_true, y_pred)
            return loss

        def heading_class_loss(self, y_true, y_pred):
            loss = self._cross_entropy(y_true, y_pred)
            return loss

        def heading_residual_loss(self, y_true, y_pred):
            loss = self._huber_loss(y_true, y_pred)
            return loss

        def size_class_loss(self, y_true, y_pred):
            loss = self._cross_entropy(y_true, y_pred)
            return loss

        def size_residual_loss(self, y_true, y_pred):
            loss = self._huber_loss(y_true, y_pred)
            return loss

        def centroid_loss(self, y_true, y_pred):
            loss = self._huber_loss(y_true, y_pred)
            return loss

        def corners_loss(self, y_true, y_pred):
            loss = self._huber_loss(y_true, y_pred)
            return loss
