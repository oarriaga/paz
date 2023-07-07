import numpy as np
import math
import tensorflow as tf

import paz.processors as pr

from paz.backend.processor.image import CastImage
from paz.abstract import Processor, SequentialProcessor

from mask_rcnn.backend.boxes import generate_pyramid_anchors
from mask_rcnn.backend.boxes import extract_boxes_from_masks
from mask_rcnn.backend.boxes import compute_RPN_bounding_box
from mask_rcnn.backend.boxes import compute_anchor_boxes_overlaps
from mask_rcnn.backend.boxes import compute_RPN_match

from mask_rcnn.backend.image import subtract_mean_image, crop_resize_masks


class DataGenerator(Processor):
    """An iterable that returns images and corresponding target class ids,
    bounding box deltas, and masks. It inherits from keras.utils.Sequence
    to avoid data redundancy when multiprocessing=True.
    dataset: The Dataset object to pick data from
    shuffle: If True, shuffles the samples before every epoch
    augmentation: Optional. From paz pipeline for augmentation.
    inputs list:
        - images: [batch, H, W, C]
        - RPN_match: [batch, N] Integer
                     (1=positive anchor, -1=negative, 0=neutral)
        - RPN_bounding_box: [batch, N, (dy, dx, log(dh), log(dw))]
                            Anchor bounding_box deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and
                    width are those of the image unless use_mini_mask is True,
                    in which case they are defined in MINI_MASK_SHAPE.
    outputs list: Usually empty in regular training. But if
                  detection_targets is True then the outputs list contains
                  target class_ids, bounding_box deltas, and masks.
    """
    def __init__(self):
        super(DataGenerator, self).__init__()

    def call(self, dataset, backbone, image_shape, anchor_scales, batch_size,
             shuffle=True, pixel_mean=np.array([123.7, 116.8, 103.9])):
        batch_num = 0
        image_index = -1

        backbone_shapes = ComputeBackboneShapes()(backbone, image_shape)
        anchors = GeneratePyramidAnchors()(anchor_scales, backbone_shapes,
                                           [0.5, 1, 2], [4, 8, 16, 32, 64])
        while True:
            # Increment index to pick next image. Shuffle if at the start of
            # an epoch.
            image_index = GetImageIndex()(image_index, len(dataset.load_data()
                                                           ), shuffle)
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, class_ids, boxes, masks = LoadImage()(dataset, image_id)
            RPN_match, RPN_bounding_box = BuildRPNTargets()(anchors, class_ids,
                                                            boxes)

            if batch_num == 0:
                batch_RPN_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=RPN_match.dtype)
                batch_RPN_bounding_box = np.zeros(
                    [batch_size, 256, 4], dtype=RPN_bounding_box.dtype)
                batch_images = np.zeros(
                    [batch_size, image_shape[0], image_shape[1],
                     image_shape[2]], dtype=np.float32)
                batch_class_ids = np.zeros(
                    (batch_size, 100), dtype=np.int32)
                batch_boxes = np.zeros(
                    (batch_size, 100, 4), dtype=np.int32)
                batch_masks = np.zeros(
                    (batch_size, masks.shape[0], masks.shape[1],
                     100), dtype=bool)

            image = CastImage('float32')(image)

            # Add to batch
            batch_RPN_match[batch_num] = RPN_match[:, np.newaxis]
            batch_RPN_bounding_box[batch_num, :RPN_bounding_box.shape[0]
                                   ] = RPN_bounding_box
            batch_images[batch_num] = subtract_mean_image(image, pixel_mean)
            batch_class_ids[batch_num, :class_ids.shape[0]] = class_ids
            batch_boxes[batch_num, :boxes.shape[0]] = boxes
            batch_masks[batch_num, :, :, :masks.shape[-1]] = masks

            batch_num += 1

            if batch_num >= batch_size:
                batch_RPN = ConcatenateRPNValues()(
                    batch_RPN_bounding_box, batch_RPN_match, batch_size,
                    batch_num, anchors.shape[0])

                inputs = [batch_images, batch_class_ids, batch_boxes,
                          batch_masks]
                outputs = [batch_RPN_match, batch_RPN]

                yield inputs, outputs
                batch_num = 0


class GetImageIndex(Processor):
    """Increment index to pick next image. Shuffle if at the start of
    an epoch.."""

    def __init__(self):
        super(GetImageIndex, self).__init__()

    def call(self, image_index, dataset_length, shuffle=True):
        image_ids = np.array([i for i in range(dataset_length)])
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)
        return image_index


class ConcatenateRPNValues(Processor):
    """Hack to handle RPN class and RPN bounding box losses."""

    def __init__(self):
        super(ConcatenateRPNValues, self).__init__()

    def call(self, batch_RPN_bounding_box, batch_RPN_match, batch_size,
             batch_num, anchors_shape):
        zeros_array = np.zeros((batch_size, anchors_shape, 3))
        RPN_match_padded = np.concatenate((batch_RPN_match, zeros_array),
                                          axis=2)
        batch_RPN_values = []
        for batch_num in range(batch_size):
            batch_RPN_values.append(np.concatenate((
                batch_RPN_bounding_box[batch_num],
                RPN_match_padded[batch_num])))

        batch_RPN_values = np.stack(batch_RPN_values, axis=0)
        return batch_RPN_values


class GeneratePyramidAnchors(Processor):
    """Computes the width and height of each stage of the backbone network.

    Arguments:
      backbone = backbone used by network
      image_shape = [height, width, channel]
    Returns:
      [N, (height, width)]. Where N is the number of stages.
    """
    def __init__(self):
        super(GeneratePyramidAnchors, self).__init__()

    def call(self, anchor_scales, backbone_shapes, anchor_ratios, strides):
        return generate_pyramid_anchors(anchor_scales, anchor_ratios,
                                        backbone_shapes, strides, 1)


class ComputeBackboneShapes(Processor):
    """Computes the width and height of each stage of the backbone network.

    Arguments:
      backbone = backbone used by network
      image_shape = [height, width, channel]
    Returns:
      [N, (height, width)]. Where N is the number of stages.
    """
    def __init__(self):
        super(ComputeBackboneShapes, self).__init__()

    def call(self, backbone, image_shape):

        if callable(backbone):
            return compute_backbone_shapes(image_shape)

        # Supports ResNet only
        assert backbone in ["resnet50", "resnet101"]
        stages = []
        for stride in [4, 8, 16, 32, 64]:
            stages.append([int(math.ceil(image_shape[0] / stride)),
                           int(math.ceil(image_shape[1] / stride))])
        return np.array(stages)


class LoadImage(Processor):
    """Load and return ground truth data for an image
    (image, mask, bounding boxes).

    # Arguments:
        dataset: The dataset object containing the image data.
        image_id: The ID of the image to load.
        use_mini_mask: A boolean flag indicating whether to use mini masks.
                        If True, the mask dimensions will be defined by
                        smaller_mask_shape. If False, the mask dimensions
                        will be the same as the image dimensions. (Default:
                        False)
        smaller_mask_shape: A tuple (height, width) specifying the
                            dimensions of the mini masks when use_mini_mask
                            is True. (Default: (28, 28))

    # Returns:
        image: [height, width, 3]
        class_ids: [instance_count] Integer class IDs
        bounding_box: [instance_count, (y1, x1, y2, x2)]
        mask: [height, width, instance_count]. The height and width are
              those of the image unless use_mini_mask is True, in which case
              they are defined in MINI_MASK_SHAPE.
        """
    def __init__(self):
        super(LoadImage, self).__init__()

    def call(self, dataset, image_id, use_mini_mask=False,
             smaller_mask_shape=(28, 28)):
        data = dataset.load_data()
        image = data[image_id]['image']
        mask = data[image_id]['masks']
        class_ids = data[image_id]['box_data'][:, -1]

        positive_mask_indices = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, positive_mask_indices]

        bounding_box = extract_boxes_from_masks(mask)

        if use_mini_mask:
            mask = generate_smaller_masks(bounding_box.astype(np.int32), mask,
                                          smaller_mask_shape)

        return image, np.array(class_ids), bounding_box, mask.astype(np.bool)


class BuildRPNTargets(Processor):
    """Given the anchors and Ground truth boxes, compute overlaps and identify
    positive anchors and deltas to refine them to match their corresponding
    Ground truth boxes.

    # Arguments:
        anchors: A numpy array of shape [num_anchors, (y1, x1, y2, x2)]
                 representing the anchor boxes.
        class_ids: A numpy array of shape [num_gt_boxes] containing integer
                   class IDs of the GT boxes.
        boxes: A numpy array of shape [num_gt_boxes, (y1, x1, y2, x2)]
               representing the GT boxes.

    # Returns:
        RPN_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        RPN_bounding_box: [N, (dy, dx, log(dh), log(dw))] Anchor bounding_box
                          deltas.
    """
    def __init__(self):
        super(BuildRPNTargets, self).__init__()

    def call(self, anchors, class_ids, boxes):
        overlaps, no_crowd_bool, boxes = BuildAnchorBoxOverlap()(
            anchors, class_ids, boxes)
        RPN_match, anchor_IoU_argmax = ComputeRPNMatch()(anchors, overlaps,
                                                         no_crowd_bool)
        RPN_bounding_box = ComputeRPNBoundingBox()(anchors, RPN_match, boxes,
                                                   anchor_IoU_argmax)
        return RPN_match, np.array(RPN_bounding_box).astype(np.float32)


class BuildAnchorBoxOverlap(Processor):
    """Given the anchors and Ground truth boxes, compute overlaps and identify
    positive anchors and deltas to refine them to match their corresponding
    Ground truth boxes.

    # Arguments:
        anchors: A numpy array of shape [num_anchors, (y1, x1, y2, x2)]
                 representing the anchor boxes.
        class_ids: A numpy array of shape [num_gt_boxes] containing integer
                   class IDs of the GT boxes.
        boxes: A numpy array of shape [num_gt_boxes, (y1, x1, y2, x2)]
               representing the GT boxes.

    # Returns:
        RPN_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        RPN_bounding_box: [N, (dy, dx, log(dh), log(dw))] Anchor bounding_box
                          deltas.
    """
    def __init__(self):
        super(BuildAnchorBoxOverlap, self).__init__()

    def call(self, anchors, class_ids, boxes):
        return compute_anchor_boxes_overlaps(anchors, class_ids, boxes)


class ComputeRPNMatch(Processor):
    """Given the anchors and Ground truth boxes, compute overlaps and identify
    positive anchors and deltas to refine them to match their corresponding
    Ground truth boxes.

    # Arguments:
        anchors: A numpy array of shape [num_anchors, (y1, x1, y2, x2)]
                 representing the anchor boxes.
        class_ids: A numpy array of shape [num_gt_boxes] containing integer
                   class IDs of the GT boxes.
        boxes: A numpy array of shape [num_gt_boxes, (y1, x1, y2, x2)]
               representing the GT boxes.

    # Returns:
        RPN_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        RPN_bounding_box: [N, (dy, dx, log(dh), log(dw))] Anchor bounding_box
                          deltas.
    """
    def __init__(self):
        super(ComputeRPNMatch, self).__init__()

    def call(self, anchors, overlaps, no_crowd_bool):
        return compute_RPN_match(anchors, overlaps, no_crowd_bool)


class ComputeRPNBoundingBox(Processor):
    """Given the anchors and Ground truth boxes, compute overlaps and identify
    positive anchors and deltas to refine them to match their corresponding
    Ground truth boxes.

    # Arguments:
        anchors: A numpy array of shape [num_anchors, (y1, x1, y2, x2)]
                 representing the anchor boxes.
        class_ids: A numpy array of shape [num_gt_boxes] containing integer
                   class IDs of the GT boxes.
        boxes: A numpy array of shape [num_gt_boxes, (y1, x1, y2, x2)]
               representing the GT boxes.

    # Returns:
        RPN_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        RPN_bounding_box: [N, (dy, dx, log(dh), log(dw))] Anchor bounding_box
                          deltas.
    """
    def __init__(self):
        super(ComputeRPNBoundingBox, self).__init__()

    def call(self, anchors, RPN_match, boxes, anchor_IoU_argmax):
        return compute_RPN_bounding_box(anchors, RPN_match, boxes,
                                        anchor_IoU_argmax)
