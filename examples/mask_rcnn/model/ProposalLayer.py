import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import context

from mask_rcnn.layer_utils import slice_batch, trim_anchors_by_score, \
    apply_box_deltas, compute_NMS, clip_image_boundaries


class ProposalLayer(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
       to the second stage. Filtering is done based on anchor scores and
       non-max suppression to remove overlaps. It also applies bounding
       box refinement deltas to anchors.

    # Arguments:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y_min, x_min, y_max, x_max)] anchors
                 in normalized coordinates

    # Returns:
        Normalized proposals [batch, rois, (y_min, x_min, y_max, x_max)]
    """

    def __init__(self, proposal_count, nms_threshold, rpn_bbox_std_dev,
                 pre_nms_limit, images_per_gpu, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.pre_nms_limit = pre_nms_limit
        self.images_per_gpu = images_per_gpu
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

    def call(self, inputs):
        scores, deltas, anchors = inputs
        scores = scores[:, :, 1]
        deltas = deltas * np.reshape(self.rpn_bbox_std_dev, [1, 1, 4])
        scores, deltas, pre_nms_anchors = trim_anchors_by_score(scores, deltas,
                                                                anchors, self.images_per_gpu,
                                                                self.pre_nms_limit)
        boxes = apply_box_deltas(pre_nms_anchors, deltas, self.images_per_gpu)
        boxes = clip_image_boundaries(boxes, self.images_per_gpu)

        proposals = slice_batch([boxes, scores], [self.proposal_count, self.nms_threshold],
                                compute_NMS, self.images_per_gpu)

        # if not context.executing_eagerly():
        #     # Infer the static output shape:
        #     out_shape = self.compute_output_shape(None)
        #     proposals.set_shape(out_shape)
        return proposals
