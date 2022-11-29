import numpy as np
import paz.processors as pr
from paz.abstract import SequentialProcessor

import necessary_imports as ni
from utils import get_class_name_efficientdet


def process_outputs(outputs):
    """Merges feature levels, boxes and classes into single tensor.

    # Arguments
        outputs: Tensor, prrior box coordinate offsets.

    # Returns
        outputs: Array, processed outputs.
    """
    outputs = outputs[0]
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = np.hsplit(boxes, 4)
    boxes = np.concatenate([s2, s1, s4, s3], axis=1)
    boxes = boxes[np.newaxis]
    classes = classes[np.newaxis]
    outputs = np.concatenate([boxes, classes], axis=2)
    return outputs


def efficientdet_postprocess(model, outputs, image_scales, raw_images=None):
    """Postprocess EfficientDet output.

    # Arguments
        model: EfficientDet model
        outputs: Tensor, prior box coordinate offsets.
        image_scales: Array, raw images resize scale.
        raw_images: Array, RGB image to draw detections.

    # Returns
        image: Array, input image with detections overlaid.
        outputs: List, holding bounding box and class details.
    """
    outputs = process_outputs(outputs)
    postprocessing = SequentialProcessor(
        [pr.Squeeze(axis=None),
         pr.DecodeBoxes(model.prior_boxes, variances=[1, 1, 1, 1]),
         ni.ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
         pr.FilterBoxes(get_class_name_efficientdet('COCO'), 0.4)])
    outputs = postprocessing(outputs)
    draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('COCO'))
    for output_box2D in outputs:
        x_min, y_min, x_max, y_max = output_box2D.coordinates
        output_box2D.coordinates = (int(x_min), int(y_min),
                                    int(x_max), int(y_max))
    image = draw_boxes2D(raw_images.astype('uint8'), outputs)
    return image, outputs
