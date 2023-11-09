import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class ClassLoss(Layer):
    """Computes loss for Mask RCNN architecture, for MRCNN class loss
    Loss for the classifier head of Mask RCNN.

    # Arguments:
        target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
                          padding to fill in the array.
        pred_class_logits: [batch, num_rois, num_classes]
        active_class_ids: [batch, num_classes]. Has a value of 1 for
                          classes that are in the dataset of the image, and 0
                          for classes that are not in the dataset.

    # Returns:
        loss: class loss value
    """

    def __init__(self, num_classes, loss_weight=1.0, name='mrcnn_class_loss',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.active_class_ids = tf.ones([num_classes], dtype=tf.int32)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        pred_class_ids = tf.argmax(input=y_pred, axis=2)

        pred_active = tf.gather(self.active_class_ids, pred_class_ids)

        y_true = tf.cast(y_true, 'int64')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                              logits=y_pred)
        loss = loss * tf.cast(pred_active, 'float32')

        loss = tf.math.reduce_sum(loss) / (tf.math.reduce_sum(
            input_tensor=tf.cast(pred_active, 'float32')))
        self.add_loss(loss * self.loss_weight)

        metric = (loss * self.loss_weight)
        self.add_metric(metric, name='mrcnn_class_loss', aggregation='mean')
        return loss
