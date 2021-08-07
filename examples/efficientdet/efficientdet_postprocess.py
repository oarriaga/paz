import numpy as np
import paz.processors as pr
from paz.abstract import SequentialProcessor
from utils import get_class_name_efficientdet


def merge_level_outputs(class_outputs, box_outputs, num_levels, num_classes):
    """
    Merges all feature levels into single tensor.

    # Arguments
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        num_levels: Int, number of levels considered at efficientnet features.
        num_classes: Int, number of classes in the dataset.

    # Returns
        class_outputs: Numpy tensor, logits for all classes corresponding to
        the features associated with the box coordinates irrespective of
        feature levels.
        box_outputs: Numpy tensor, box coordinate offsets for the corresponding
        prior boxes irrespective of feature levels.
    """
    class_outputs_all, box_outputs_all = [], []
    batch_size = class_outputs[0].shape[0]
    for level in range(0, num_levels):
        class_out = class_outputs[level].numpy()
        class_out = class_out.reshape(batch_size, -1, num_classes)
        class_outputs_all.append(class_out)
        box_out = box_outputs[level].numpy()
        box_out = box_out.reshape(batch_size, -1, 4)
        box_outputs_all.append(box_out)
    class_outputs_all = np.concatenate(class_outputs_all, 1)
    box_outputs_all = np.concatenate(box_outputs_all, 1)
    return class_outputs_all, box_outputs_all


def process_outputs(class_outputs, box_outputs, num_levels, num_classes):
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
    class_outputs, box_outputs = merge_level_outputs(
        class_outputs, box_outputs, num_levels, num_classes)
    s1, s2, s3, s4 = np.hsplit(box_outputs[0], 4)
    box_outputs = np.concatenate([s2, s1, s4, s3], axis=1)
    box_outputs = box_outputs[np.newaxis]
    class_outputs = 1 / (1 + np.exp(-class_outputs))
    outputs = np.concatenate([box_outputs, class_outputs], axis=2)
    return outputs


def efficientdet_postprocess(model, class_outputs, box_outputs,
                             image_scales, raw_images=None):
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
    outputs = process_outputs(
        class_outputs, box_outputs, model.num_levels, model.num_classes)
    postprocessing = SequentialProcessor(
        [pr.Squeeze(axis=None),
         pr.DecodeBoxes(model.prior_boxes, variances=[1, 1, 1, 1]),
         pr.ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
         pr.FilterBoxes(get_class_name_efficientdet('COCO'), 0.4)])
    outputs = postprocessing(outputs)
    draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('COCO'))
    image = draw_boxes2D(raw_images.astype('uint8'), outputs)
    return image, outputs
