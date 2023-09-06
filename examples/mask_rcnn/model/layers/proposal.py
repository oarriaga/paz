import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import context

from mask_rcnn.model.layer_utils import apply_box_delta, slice_batch
from mask_rcnn.model.layer_utils import clip_boxes


def trim_anchors(scores, deltas, anchors, images_per_gpu, pre_nms_limit):
    """Selects fixed number of anchors before NMS.

    # Arguments:
        scores: [N] Predicted target class values.
        deltas: [N, (dy, dx, log(dw), log(dh))] refinements to apply.
        anchors: [batch, num_anchors, (x_min, y_min, x_max, y_max)] anchors
                 in normalized coordinates.
        images_per_gpu: Number of images to train with on each GPU.
        pre_nms_limit: type int, ROIs kept to keep
                       before non-maximum suppression.
    # Returns:
        scores: [pre_nms_limit] Predicted target class values.
        deltas: [pre_nms_limit, (dx, dy, log(dw), log(dh))] refinements to
                apply.
        anchors: [batch, pre_nms_limit, (x_min, y_min, x_max, y_max)].
    """
    pre_nms_limit = tf.minimum(pre_nms_limit, tf.shape(anchors)[1])
    indices = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                          name='top_anchors').indices
    scores = slice_batch([scores, indices], [], tf.gather, images_per_gpu)
    deltas = slice_batch([deltas, indices], [], tf.gather, images_per_gpu)
    pre_nms_anchors = slice_batch([anchors, indices], [], tf.gather,
                                  images_per_gpu, names=['pre_nms_anchors'])
    return scores, deltas, pre_nms_anchors


def apply_box_deltas(pre_nms_anchors, deltas, images_per_gpu):
    """Applies refinement to anchors based on delta values in slices.

    # Arguments:
        pre_nms_anchors: [N, (x_min, y_min, x_max, y_max)] boxes to update.
        deltas: [N, (dx, dy, log(dw), log(dh))] refinements to apply.
        images_per_gpu: Number of images to train with on each GPU.
    # Returns:
        boxes: [N, (x_min, y_min, x_max, y_max)].
    """
    boxes = slice_batch([pre_nms_anchors, deltas], [], apply_box_delta,
                        images_per_gpu, names=['refined_anchors'])
    return boxes


def clip_image_boundaries(boxes, images_per_gpu):
    """Clips boxes to the given window size in this case the normalised
    image boundaries.

    # Arguments:
        boxes: [N, (dx, dy, log(dw), log(dh))] refinements to apply.
        images_per_gpu: Number of images to train with on each GPU.
    # Returns:
        boxes: [N, (dx, dy, log(dw), log(dh))].
    """
    size = np.array([0, 0, 1, 1], dtype=np.float32)
    boxes = slice_batch(boxes, [size], clip_boxes, images_per_gpu,
                        names=['refined_anchors_clipped'])
    return boxes


def compute_NMS(boxes, scores, proposal_count, nms_threshold):
    """Computes non-max suppression on the image and refining the shape of
    the proposals.

    # Arguments:
        boxes: [N, (dx, dy, log(dw), log(dh))] refinements to apply.
        scores: [N] Predicted target class values.
        proposal_count: Max number of proposals.
        nms_threshold: Non-maximum suppression threshold for detection.
    # Returns:
        proposals: Normalized proposals
                   [batch, N, (x_min, y_min, x_max, y_max)]
    """
    indices = tf.image.non_max_suppression(
        boxes, scores, proposal_count, nms_threshold,
        name='rpn_non_max_suppression')
    proposals = tf.gather(boxes, indices)
    proposals_shape = tf.shape(proposals)[0]
    padding = tf.maximum(proposal_count - proposals_shape, 0)
    proposals = tf.pad(proposals, [(0, padding), (0, 0)])
    return proposals


class ProposalLayer(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
       to the second stage. Filtering is done based on anchor scores and
       non-max suppression to remove overlaps. It also applies bounding
       box refinement deltas to anchors.

    # Arguments:
        RPN_probs: [batch, num_anchors, (bg prob, fg prob)].
        RPN_bounding_box: [batch, num_anchors, (dx, dy, log(dw), log(dh))].
        anchors: [batch, num_anchors, (x_min, y_min, x_max, y_max)] anchors
                 in normalized coordinates.

    # Returns:
        Normalized proposals [batch, ROIs, (x_min, y_min, x_max, y_max)]
    """

    def __init__(self, proposal_count, nms_threshold, RPN_bounding_box_std_dev,
                 pre_nms_limit, images_per_gpu, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.RPN_bounding_box_std_dev = RPN_bounding_box_std_dev
        self.pre_nms_limit = pre_nms_limit
        self.images_per_gpu = images_per_gpu
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

    def call(self, inputs):
        scores, deltas, anchors = inputs
        scores = scores[:, :, 1]
        deltas = deltas * np.reshape(self.RPN_bounding_box_std_dev, [1, 1, 4])
        scores, deltas, pre_nms_anchors = trim_anchors(scores, deltas, anchors,
                                                       self.images_per_gpu,
                                                       self.pre_nms_limit)

        boxes = apply_box_deltas(pre_nms_anchors, deltas, self.images_per_gpu)
        boxes = clip_image_boundaries(boxes, self.images_per_gpu)

        proposals = slice_batch([boxes, scores],
                                [self.proposal_count, self.nms_threshold],
                                compute_NMS, self.images_per_gpu)
        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = (None, self.proposal_count, 4)
            proposals.set_shape(out_shape)
        return proposals
