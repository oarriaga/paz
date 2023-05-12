import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer


class ProposalClassLoss(tf.keras.losses.Loss):
    """Computes loss for Mask RCNN architecture, Region Proposal
    Network Class loss. RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
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
