import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer


class MaskLoss(Layer):
    """Computes loss for Mask RCNN architecture, for MRCNN mask loss.
    Mask binary cross-entropy loss for the masks head.

    # Arguments:
        target_masks: [batch, num_ROIs, height, width].
                      A float32 tensor of values 0 or 1. Uses zero padding to
                      fill array.
        target_class_ids: [batch, num_ROIs]. Integer class IDs. Zero padded
        pred_masks: [batch, proposals, height, width, num_classes] float32
                    tensor with values from 0 to 1.

    # Returns:
        loss: mask loss
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_mask_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        true_masks, target_ids, pred_masks = resize_targets(y_true, y_pred)

        positive_indices = tf.where(target_ids > 0)[:, 0]
        positive_class_ids = tf.cast(tf.gather(target_ids, positive_indices),
                                     tf.int64)

        indices = tf.stack([positive_indices, positive_class_ids], axis=1)

        y_true = tf.gather(true_masks, positive_indices)
        y_pred = tf.gather_nd(pred_masks, indices)

        loss = K.switch(tf.size(y_true) > 0,
                        K.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
        loss = K.mean(loss)
        self.add_loss(loss * self.loss_weight)

        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_mask_loss', aggregation='mean')
        return loss


def resize_target(target, size):
    """Reshapes the y_pred, ground truth class ids and masks.

    # Arguments:
        target : Target value to be resized
        shape: size to resize the target into

    # Returns:
        resized_target
    """
    resized_target = K.reshape(target, size)
    return resized_target


def resize_targets(y_true, y_pred):
    """Reshapes the y_pred, ground truth class ids and masks.

        # Arguments:
            target_masks: [batch, num_ROIs, height, width].
                          A float32 tensor of values 0 or 1. Uses zero padding
                          to fill array.
            target_class_ids: [batch, num_ROIs]. Integer class IDs.
            pred_masks: [batch, proposals, height, width, num_classes]
                        float32 tensor with values from 0 to 1.

        # Returns:
            target_ids: [batch * num_ROIs]. Integer class IDs. Reshaped
            target_masks: [batch * num_ROIs, height, width].
                          A float32 tensor of values 0 or 1. Reshaped
            y_pred: [batch * proposals, num_classes, height, width] float32
                    tensor with values from 0 to 1. Reshaped and transposed
        """
    mask_shape = tf.shape(y_true[0])
    target_masks = resize_target(y_true[0], (-1, mask_shape[2], mask_shape[3]))
    target_ids = resize_target(y_true[1], (-1,))

    pred_shape = tf.shape(y_pred)
    y_pred = resize_target(y_pred, (-1, pred_shape[2], pred_shape[3],
                                    pred_shape[4]))

    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])
    return target_masks, target_ids, y_pred
