import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer

from mask_rcnn.model.model_utils import smooth_L1_loss


class BBoxLoss(Layer):
    """Computes loss for Mask RCNN architecture, MRCNN BBox loss
    Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    def __init__(self, loss_weight=1.0, name='mrcnn_bbox_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        target_boxes = y_true[0]
        target_class_ids = y_true[1]
        target_class_ids = K.reshape(target_class_ids, (-1,))
        target_boxes = K.reshape(target_boxes, (-1, 4))
        predicted_boxes = K.reshape(y_pred,
                                    (-1, K.int_shape(y_pred)[2], 4))
        positive_ROI_indices = tf.where(target_class_ids > 0)[:, 0]
        positive_ROI_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ROI_indices), tf.int64)
        indices = tf.stack([positive_ROI_indices, positive_ROI_class_ids],
                           axis=1)
        target_boxes = tf.gather(target_boxes, positive_ROI_indices)
        predicted_boxes = tf.gather_nd(predicted_boxes, indices)
        loss = K.switch(tf.size(target_boxes) > 0,
                        smooth_L1_loss(target_boxes, predicted_boxes),
                        tf.constant(0.0))
        loss = K.mean(loss)

        self.add_loss(loss * self.loss_weight)
        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_bbox_loss', aggregation='mean')
        return loss
