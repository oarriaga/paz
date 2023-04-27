import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.eager import context

from mask_rcnn.utils import norm_boxes_graph
from mask_rcnn.layer_utils import slice_batch, refine_detections


class DetectionLayer(Layer):
    """Detects final bounding boxes and masks for given proposals

    # Arguments:

    # Returns:
        [batch, num_detections, (y_min, x_min, y_max, x_max, class_id,
         class_score)] where coordinates are normalized.
    """

    def __init__(self, batch_size, window, bbox_std_dev, images_per_gpu, detection_max_instances,
                 detection_min_confidence, detection_nms_threshold, image_shape,**kwargs,):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.window = window
        self.bbox_std_dev = bbox_std_dev
        self.images_per_gpu = images_per_gpu
        self.detection_max_instances = detection_max_instances
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_threshold = detection_nms_threshold
        self.image_shape = image_shape

    def call(self, inputs):
        rois, mrcnn_class, mrcnn_bbox = inputs
        # self.window = norm_boxes_graph(self.window, self.image_shape[:2])
        detections_batch = slice_batch([rois, mrcnn_class, mrcnn_bbox],
                                       [tf.cast(self.bbox_std_dev, dtype=tf.float32),
                                        self.window, self.detection_min_confidence,
                                        self.detection_max_instances,
                                        tf.cast(self.detection_nms_threshold, dtype=tf.float32)],
                                       refine_detections, self.images_per_gpu)

        return tf.reshape(detections_batch,
                          [self.batch_size, self.detection_max_instances, 6])
