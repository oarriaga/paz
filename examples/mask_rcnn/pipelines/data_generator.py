import numpy as np
import math
import tensorflow as tf

from paz.processors.image import CastImage, SubtractMeanImage
from paz.processors.standard import SequenceWrapper, ControlMap
from paz.processors.standard import UnpackDictionary
from paz.abstract import Processor, SequentialProcessor

from mask_rcnn.backend.boxes import generate_pyramid_anchors
from mask_rcnn.backend.boxes import compute_RPN_bounding_box
from mask_rcnn.backend.boxes import compute_anchor_boxes_overlaps
from mask_rcnn.backend.boxes import compute_RPN_match, concatenate_RPN_values

from mask_rcnn.backend.image import subtract_mean_image, crop_resize_masks


class MaskRCNNPipeline(SequentialProcessor):
    def __init__(self, image_shape, anchor_scales, backbone,
                 pixel_mean=np.array([123.7, 116.8, 103.9]), max_instances=100,
                 anchor_size=256, strides=np.array([4, 8, 16, 32, 64]),
                 ROI_positive_ratio=0.33):
        super(MaskRCNNPipeline, self).__init__()
        self.make_rpn_labels = MakeRPNLabel(image_shape, anchor_scales,
                                            backbone)
        self.preprocess_data = PreprocessImages(pixel_mean)

        self.add(UnpackDictionary(['input_image', 'input_gt_class_ids',
                                   'input_gt_boxes', 'input_gt_masks']))
        self.add(ControlMap(self.make_rpn_labels, [1, 2], [1, 2, 4, 5]))
        self.add(ControlMap(self.preprocess_data, [0, 1, 2, 3], [0, 1, 2, 3]))

        rpn_size = int(np.sum(strides) * ROI_positive_ratio * max_instances)

        self.add(SequenceWrapper(
            {0: {'input_image': [image_shape[0], image_shape[1],
                                 image_shape[2]]},
             1: {'input_gt_class_ids': [max_instances]},
             2: {'input_gt_boxes': [max_instances, 4]},
             3: {'input_gt_masks': [image_shape[0], image_shape[1],
                                    max_instances]}},
            {4: {'rpn_class_logits': [rpn_size, 1]},
             5: {'rpn_bounding_box': [rpn_size + anchor_size, 4]}}))


class MakeRPNLabel(Processor):
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
    def __init__(self, image_shape, anchor_scales, backbone):
        super(MakeRPNLabel, self).__init__()
        self.image_shape = image_shape

        backbone_shapes = ComputeBackboneShapes()(backbone, self.image_shape)
        self.anchors = GeneratePyramidAnchors()(
            anchor_scales, backbone_shapes, [0.5, 1, 2], [4, 8, 16, 32, 64])
        self.build_RPN_targets = BuildRPNTargets()

    def call(self, class_ids, boxes):
        RPN_match, RPN_bounding_box = self.build_RPN_targets(
            self.anchors, class_ids, boxes)
        RPN_val = concatenate_RPN_values(RPN_bounding_box, RPN_match,
                                         self.anchors.shape[0])
        RPN_match = np.reshape(RPN_match, (len(RPN_match), 1))
        return class_ids, boxes, RPN_match, RPN_val


class PreprocessImages(Processor):
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
    def __init__(self, pixel_mean=np.array([123.7, 116.8, 103.9])):
        super(PreprocessImages, self).__init__()
        self.cast_image = CastImage('float32')
        self.subtract_mean_image = SubtractMeanImage(pixel_mean)

    def call(self, image, class_ids, boxes, masks):
        image = self.cast_image(image)
        image = self.subtract_mean_image(image)

        boxes_zeros = np.zeros((100, 4))
        class_ids_zeros = np.zeros(100)
        masks_zeros = np.zeros((128, 128, 100))
        boxes_zeros[:len(boxes)] = boxes
        class_ids_zeros[:len(class_ids)] = class_ids
        for i in range(np.shape(masks)[2]):
            masks_zeros[:, :, i] = masks[:, :, i]
        return image, class_ids_zeros, boxes_zeros, masks_zeros


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
        overlaps, no_crowd_bool, boxes = compute_anchor_boxes_overlaps(
            anchors, class_ids, boxes)
        RPN_match, anchor_IoU_argmax = compute_RPN_match(
            anchors, overlaps, no_crowd_bool)
        RPN_bounding_box = compute_RPN_bounding_box(
            anchors, RPN_match, boxes, anchor_IoU_argmax)
        return RPN_match, np.array(RPN_bounding_box).astype(np.float32)


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
            return ComputeBackboneShapes()(backbone, image_shape)

        # Supports ResNet only
        assert backbone in ["resnet50", "resnet101"]
        stages = []
        for stride in [4, 8, 16, 32, 64]:
            stages.append([int(math.ceil(image_shape[0] / stride)),
                           int(math.ceil(image_shape[1] / stride))])
        return np.array(stages)


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
