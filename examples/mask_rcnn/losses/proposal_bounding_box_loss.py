import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Layer


class ProposalBoundingBoxLoss(tf.keras.losses.Loss):
    """ Computes the loss for the Mask RCNN architecture for the Region
    Proposal Network (RPN) bounding box loss. This loss function calculates
    the RPN bounding box loss graph.

    Arguments:
        anchors_per_image: An integer indicating the number of anchors per
                           image.
        image_per_gpu: An integer indicating the number of images per GPU.
        loss_weight: A float specifying the loss weight (default: 1.0).
        y_true: The ground truth tensor containing the target bounding boxes.
                    Shape: [batch, max positive anchors,
                    (dy, dx, log(dh), log(dw))]. Uses 0 padding to fill in
                    unused bounding box deltas.
        y_pred: The predicted tensor containing the RPN bounding boxes.
                    Shape: [batch, anchors, (dy, dx, log(dh), log(dw))].

    Returns:
        The computed loss value
    """

    def __init__(self, anchors_per_image, image_per_gpu, loss_weight=1.0,
                 name='rpn_bounding_box_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.anchors_per_image = anchors_per_image
        self.images_per_gpu = image_per_gpu
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        input_rpn_bounding_box = y_true[:, :self.anchors_per_image, :]
        rpn_match = y_true[:, self.anchors_per_image:, :1]

        rpn_match = K.squeeze(rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))

        rpn_bounding_box = tf.gather_nd(y_pred, indices)
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_boxes = batch_pack_graph(input_rpn_bounding_box, batch_counts,
                                        self.images_per_gpu)

        loss = smooth_L1_loss(target_boxes, rpn_bounding_box)
        loss = K.switch(tf.size(input=loss) > 0, K.mean(loss),
                        tf.constant(0.0))

        return loss


def batch_pack_graph(boxes, counts, num_rows):
    """Packs the boxes into a single tensor based on the counts per row.

        Arguments:
            boxes: A tensor of shape [num_rows, max_counts, ...] representing
                   the boxes.
            counts: A tensor of shape [num_rows] representing the counts per
                    row.
            num_rows: An integer indicating the number of rows.

        Returns:
            A tensor representing the packed boxes.
    """
    outputs = []
    for row in range(num_rows):
        outputs.append(boxes[row, :counts[row]])
    return tf.concat(outputs, axis=0)


def smooth_L1_loss(y_true, y_pred):
    """Calculates the smooth L1 loss between the true and predicted values.

        Arguments:
            y_true: The true tensor.
            y_pred: The predicted tensor.

        Returns:
            The computed smooth L1 loss.
    """
    diff = tf.math.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), 'float32')
    loss = (less_than_one * 0.5 * diff ** 2) + \
           (1 - less_than_one) * (diff - 0.5)
    return loss
