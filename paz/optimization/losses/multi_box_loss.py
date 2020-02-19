import tensorflow.keras.backend as K
import tensorflow as tf


class MultiboxLoss(object):
    def __init__(self, neg_pos_ratio=3, alpha=1.0, max_num_negatives=300):

        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.max_num_negatives = max_num_negatives

    def smooth_l1(self, y_true, y_pred):
        absolute_value_loss = K.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.less(absolute_value_loss, 1.0)
        l1_smooth_loss = tf.where(absolute_value_condition, square_loss,
                                  absolute_value_loss - 0.5)
        return K.sum(l1_smooth_loss, axis=-1)

    def cross_entropy(self, y_true, y_pred):
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        cross_entropy_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return cross_entropy_loss

    def compute_loss(self, y_true, y_pred):

        batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)
        class_loss = self.cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
        local_loss = self.smooth_l1(y_true[:, :, :4], y_pred[:, :, :4])
        negative_mask = y_true[:, :, 4]

        positive_mask = 1.0 - negative_mask

        # calculating the positive loss for every prior box
        positive_local_losses = local_loss * positive_mask
        positive_class_losses = class_loss * positive_mask
        # calculating the positive loss per sample
        positive_class_loss = K.sum(positive_class_losses, axis=-1)
        positive_local_loss = K.sum(positive_local_losses, axis=-1)

        # obtaining the number of negatives in the batch per sample
        num_positives_per_sample = K.cast(K.sum(positive_mask, -1), 'int32')
        num_hard_negatives = self.neg_pos_ratio * num_positives_per_sample
        num_negatives_per_sample = K.minimum(num_hard_negatives,
                                             self.max_num_negatives)
        negative_class_losses = class_loss * negative_mask
        elements = (negative_class_losses, num_negatives_per_sample)
        negative_class_loss = tf.map_fn(
            lambda x: K.sum(tf.nn.top_k(x[0], x[1])[0]),
            elements, dtype=tf.float32)
        class_loss = positive_class_loss + negative_class_loss
        total_loss = class_loss + (self.alpha * positive_local_loss)
        num_positives = K.sum(K.cast(positive_mask, 'float32'))
        return (total_loss * batch_size) / tf.maximum(1.0, num_positives)
