import numpy as np
import paz.processors as pr
from paz.abstract import SequentialProcessor
from utils import get_class_name_efficientdet


def process_outputs(outputs):
    """
    Merges all feature levels into single tensor and combines box offsets
    and class scores.

    # Arguments
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        num_levels: Int, number of levels considered at efficientnet features.
        num_classes: Int, number of classes in the dataset.

    # Returns
        outputs: Numpy array, Processed outputs by merging the features at
        all levels. Each row corresponds to box coordinate offsets and
        sigmoid of the class logits.
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
    """
    EfficientDet output postprocessing function.
    # Arguments
        model: EfficientDet model
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        image_scale: Numpy array, scale to reconstruct each of the raw images
        to original size from the resized image.
        raw_images: Numpy array, RGB image to draw the detections on the image.

    # Returns
        image: Numpy array, RGB input image with detections overlaid.
        outputs: List of Box2D, containing the detections with bounding box
        and class details.
    """
    outputs = process_outputs(outputs)
    postprocessing = SequentialProcessor(
        [pr.Squeeze(axis=None),
         pr.DecodeBoxes(model.prior_boxes, variances=[1, 1, 1, 1]),
         pr.ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
         pr.FilterBoxes(get_class_name_efficientdet('COCO'), 0.4)])
    outputs = postprocessing(outputs)
    draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('COCO'))
    image = draw_boxes2D(raw_images.astype('uint8'), outputs)
    return image, outputs
