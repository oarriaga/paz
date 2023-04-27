import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer

from mask_rcnn.model.model_utils import reshape_data


class MaskLoss(Layer):
    """Computes loss for Mask RCNN architecture, for MRCNN mask loss.
    Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_mask_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        true_masks = y_true[0]
        target_class_ids = y_true[1]
        target_ids, true_masks, pred_masks = reshape_data(target_class_ids,
                                                          true_masks, y_pred)
        positive_indices = tf.where(target_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_ids, positive_indices), tf.int64)
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
