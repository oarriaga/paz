import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer


class ProposalClassLoss(tf.keras.losses.Loss):
    """Computes the loss for the Mask RCNN architecture's Region Proposal
    Network (RPN) class loss. This loss function calculates the RPN anchor
    classifier loss.

    Args:
        loss_weight: A float specifying the loss weight (default: 1.0)
        y_true: The ground truth tensor containing the anchor match type.
                Shape: [batch, anchors, 1]. Anchor match type. 1=positive,
                -1=negative, 0=neutral anchor.
        y_pred: The predicted tensor containing the RPN classifier logits for
                BG/FG. Shape: [batch, anchors, 2].

    Returns:
        The computed loss value
    """

    def __init__(self, loss_weight=1.0, name='rpn_class_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, -1)
        anchor_class = K.cast(K.equal(y_true, 1), tf.int32)
        indices = tf.compat.v1.where(K.not_equal(y_true, 0))
        y_pred = tf.gather_nd(y_pred, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                                 output=y_pred,
                                                 from_logits=True)
        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss
