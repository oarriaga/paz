import paz.processors as pr
from paz.abstract import Processor, SequentialProcessor
from paz.pipelines import AugmentImage, PreprocessImage
from paz.pipelines import AugmentBoxes
from mask_rcnn.utils import build_rpn_targets


class MatchRPNBoxes(Processor):
    def __init__(self, anchors, config):
        super(MatchRPNBoxes, self).__init__()
        self.config = config
        self.anchors = anchors
        self.image_shape = config.IMAGE_SHAPE

    def call(self, boxes):
        class_ids = boxes[:, 4]
        matches, box_deltas = build_rpn_targets(self.image_shape,
                                                self.anchors, class_ids,
                                                boxes, self.config)
        return box_deltas, matches


class PreprocessMask(SequentialProcessor):
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessMask, self).__init__()
        self.add(pr.ResizeImage(shape))


class ComputeBoxDeltas(SequentialProcessor):
    def __init__(self, config, prior_boxes):
        super(ComputeBoxDeltas, self).__init__()
        self.add(MatchRPNBoxes(prior_boxes, config),)


class DetectionPipeline(SequentialProcessor):
    def __init__(self, config, prior_boxes, split='TRAIN', num_classes=4,
                 size=320, mean=pr.BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[.1, .2]):
        super(DetectionPipeline, self).__init__()

        self.augment_image = AugmentImage()
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        self.augment_boxes = AugmentBoxes()
        self.compute_box_deltas = ComputeBoxDeltas(config, prior_boxes)
        self.num_anchors_per_image = config.RPN_TRAIN_ANCHORS_PER_IMAGE

        self.preprocess_mask = PreprocessMask((size, size))

        self.add(pr.UnpackDictionary(['image', 'boxes', 'mask']))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.preprocess_boxes, [0, 1], [1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.compute_box_deltas, [1], [1, 2], {1: 3}))
        self.add(pr.ControlMap(self.preprocess_mask, [4], [4]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'box_deltas': [self.num_anchors_per_image, 4]},
             2: {'matches': [len(prior_boxes)]},
             3: {'boxes': [(num_classes - 1), 5]},
             4: {'mask': [size, size, (num_classes - 1)]}}))
