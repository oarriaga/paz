import paz.processors as pr
from paz.abstract import Processor, SequentialProcessor
from paz.pipelines import AugmentImage, PreprocessImage
from paz.pipelines import AugmentBoxes, PreprocessBoxes
from mask_rcnn.utils import build_rpn_targets


class MatchRPNBoxes(Processor):
    def __init__(self, config):
        super(MatchRPNBoxes, self).__init__()
        self.config = config
        self.image_shape = config.IMAGE_SHAPE

    def call(self, boxes, anchors):
        matches, boxes = build_rpn_targets(self.image_shape,
                                           anchors, boxes, self.config)
        return boxes


class PreprocessMask(SequentialProcessor):
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessMask, self).__init__()
        self.add(pr.ResizeImage(shape))


class DetectionPipeline(SequentialProcessor):
    def __init__(self, prior_boxes, split='TRAIN', num_classes=4, size=320, 
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5, variances=[.1, .2]):
        super(DetectionPipeline, self).__init__()

        self.augment_image = AugmentImage()
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        self.preprocess_mask = PreprocessMask((size, size))
        
        self.add(pr.UnpackDictionary(['image', 'boxes', 'mask']))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.preprocess_boxes, [0, 1], [1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.ControlMap(self.preprocess_mask, [2], [2]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]},
             2: {'mask': [size, size, (num_classes - 1)]}}))

