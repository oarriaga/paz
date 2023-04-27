import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer

from mask_rcnn.model.model_utils import smooth_L1_loss


class ProposalBBoxLoss(tf.keras.losses.Loss):
    """Computes loss for Mask RCNN architecture for Region Proposal
     Network Bounding box loss.
     Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    def __init__(self, anchors_per_image, image_per_gpu, loss_weight=1.0, name='rpn_bbox_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors_per_image = anchors_per_image
        self.images_per_gpu = image_per_gpu
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        input_rpn_bbox = y_true[:, :self.anchors_per_image, :]
        rpn_match = y_true[:, self.anchors_per_image:, :1]

        rpn_match = K.squeeze(rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))

        rpn_bbox = tf.gather_nd(y_pred, indices)
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_boxes = batch_pack_graph(input_rpn_bbox, batch_counts,
                                        self.images_per_gpu)
        loss = smooth_L1_loss(target_boxes, rpn_bbox)
        loss = K.switch(tf.size(input=loss) > 0, K.mean(loss), tf.constant(0.0))

        return loss
