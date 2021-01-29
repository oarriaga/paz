import numpy as np
import tensorflow as tf

from paz.models.detection.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, g_mean_size_arr, \
    get_box3d_corners_helper, get_box3d_corners


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(losses)


def FPointNet_loss(args, corner_loss_weight=10.0, box_loss_weight=1.0, mask_weight=1.0):
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
        - [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/abs/1711.08488)
    """
    mask_label, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label, \
    end_points = args[0], args[1], args[2], args[3], args[4], args[5], args[6]

    mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['mask_logits'],
                                                                              labels=tf.cast(mask_label, tf.int64)))

    center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
    stage1_center_dist = tf.norm(center_label - end_points['stage1_center'], axis=-1)
    stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

    heading_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=end_points['heading_scores'], labels=tf.cast(heading_class_label, tf.int64)))

    hcls_onehot = tf.one_hot(tf.cast(heading_class_label, tf.int64), depth=NUM_HEADING_BIN, on_value=1, off_value=0,
                             axis=-1)  # BxNUM_HEADING_BIN
    heading_residual_normalized_label = heading_residual_label / (np.pi / NUM_HEADING_BIN)
    heading_residual_normalized_loss = huber_loss(tf.reduce_sum(
        end_points['heading_residuals_normalized'] * tf.cast(hcls_onehot, dtype=tf.float32), axis=1) - \
                                                  heading_residual_normalized_label, delta=1.0)

    size_class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=end_points['size_scores'], labels=tf.cast(size_class_label, tf.int64)))

    scls_onehot = tf.one_hot(tf.cast(size_class_label, tf.int64),
                             depth=NUM_SIZE_CLUSTER,
                             on_value=1, off_value=0, axis=-1)  # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims(tf.cast(scls_onehot, dtype=tf.float32), -1),
                                [1, 1, 3])  # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum(end_points['size_residuals_normalized'] *
                                                       scls_onehot_tiled, axis=[1])  # Bx3

    mean_size_arr_expand = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # 1xNUM_SIZE_CLUSTERx3
    mean_size_label = tf.reduce_sum(scls_onehot_tiled * mean_size_arr_expand, axis=[1])  # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

    corners_3d = get_box3d_corners(end_points['center'],
                                   end_points['heading_residuals'],
                                   end_points['size_residuals'])  # (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1, 1, NUM_SIZE_CLUSTER]) * \
              tf.tile(tf.expand_dims(scls_onehot, 1), [1, NUM_HEADING_BIN, 1])  # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum(tf.cast(tf.expand_dims(tf.expand_dims(gt_mask, -1), -1), dtype=tf.float32) *
                                    corners_3d, axis=[1, 2])  # (B,8,3)

    heading_bin_centers = tf.constant(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN), dtype=tf.float32)  # (NH,)
    heading_label = tf.expand_dims(heading_residual_label, 1) + tf.expand_dims(heading_bin_centers, 0)  # (B,NH)
    heading_label = tf.reduce_sum(tf.cast(hcls_onehot, dtype=tf.float32) * heading_label, 1)
    mean_sizes = tf.expand_dims(tf.constant(g_mean_size_arr, dtype=tf.float32), 0)  # (1,NS,3)
    size_label = mean_sizes + tf.expand_dims(size_residual_label, 1)  # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(tf.expand_dims(tf.cast(scls_onehot, dtype=tf.float32), -1) * size_label,
                               axis=[1])  # (B,3)
    corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)  # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1),
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)

    total_loss = mask_loss * mask_weight + box_loss_weight * (center_loss + heading_class_loss + size_class_loss +
                                                              heading_residual_normalized_loss * 20 +
                                                              size_residual_normalized_loss * 20 +
                                                              stage1_center_loss +
                                                              corner_loss_weight * corners_loss)

    return total_loss
