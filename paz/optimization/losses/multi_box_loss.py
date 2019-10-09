import tensorflow as tf
import tensorflow.keras.backend as K


class MultiboxLoss(object):
    """Multibox loss from the Single-shot multibox detector [1]
    # Arguments
        neg_pos_ratio: Integer. Number of negative boxes with respect to
            positive losses to use for the calculation of the
            negative_crossentropy loss.
        alpha: Float. Balance between localization and classification losses.
        max_negatives: Integer. Maximum number of negative samples for
            calculating the negative crossentropy.
    # References
        [1]  (SSD) https://arxiv.org/abs/1512.02325
    """
    def __init__(self, neg_pos_ratio=3, alpha=1.0, max_negatives=300):
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.max_negatives = max_negatives

    def smooth_L1(self, y_true, y_pred):
        absolute_value_loss = K.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        absolute_value_condition = K.less(absolute_value_loss, 1.0)
        smooth_L1_loss = tf.where(
            absolute_value_condition, square_loss, absolute_value_loss - 0.5)
        return K.sum(smooth_L1_loss, axis=-1)

    def cross_entropy(self, y_true, y_pred):
        y_pred = K.maximum(K.minimum(y_pred, 1 - 1e-15), 1e-15)
        cross_entropy_loss = - K.sum(y_true * K.log(y_pred), axis=-1)
        return cross_entropy_loss

    def localization_loss(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
        local_loss = self.smooth_L1(y_true[:, :, :4], y_pred[:, :, :4])
        negative_mask = y_true[:, :, 4]
        positive_mask = 1.0 - negative_mask
        positive_local_losses = local_loss * positive_mask
        positive_local_loss = tf.reduce_sum(positive_local_losses, axis=-1)
        num_positives = tf.reduce_sum(tf.cast(positive_mask, 'float32'))
        return ((self.alpha * positive_local_loss * batch_size) /
                tf.maximum(1.0, num_positives))

    def positive_crossentropy(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
        class_loss = self.cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
        negative_mask = y_true[:, :, 4]
        positive_mask = 1 - negative_mask
        positive_class_losses = class_loss * positive_mask
        positive_class_loss = K.sum(positive_class_losses, axis=-1)
        num_positives = K.sum(K.cast(positive_mask, 'float32'))
        return ((positive_class_loss * batch_size) /
                tf.maximum(1.0, num_positives))

    def negative_crossentropy(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
        class_loss = self.cross_entropy(y_true[:, :, 4:], y_pred[:, :, 4:])
        negative_mask = y_true[:, :, 4]
        positive_mask = 1 - negative_mask
        # obtaining the number of negatives in the batch per sample
        num_positives_per_sample = K.cast(K.sum(positive_mask, -1), 'int32')
        num_hard_negatives = self.neg_pos_ratio * num_positives_per_sample
        num_negatives_per_sample = K.minimum(num_hard_negatives,
                                             self.max_negatives)
        negative_class_losses = class_loss * negative_mask
        elements = (negative_class_losses, num_negatives_per_sample)
        negative_class_loss = tf.map_fn(
            lambda x: K.sum(tf.nn.top_k(x[0], x[1])[0]),
            elements, dtype=tf.float32)
        num_positives = K.sum(K.cast(positive_mask, 'float32'))
        return ((negative_class_loss * batch_size) /
                tf.maximum(1.0, num_positives))

    def __call__(self, y_true, y_pred, sample_weight=None):
        smooth_L1_loss = self.localization_loss(y_true, y_pred)
        positive_crossentropy = self.positive_crossentropy(y_true, y_pred)
        negative_crossentropy = self.negative_crossentropy(y_true, y_pred)
        return smooth_L1_loss + positive_crossentropy + negative_crossentropy
