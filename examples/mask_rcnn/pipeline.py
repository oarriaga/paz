import paz.processors as pr
from paz.abstract import Processor
from paz.abstract import SequentialProcessor
from paz.abstract.sequence import SequenceExtra
from paz.pipelines.image import AugmentImage
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


class PreprocessImage(SequentialProcessor):
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        if mean is None:
            self.add(pr.NormalizeImage())
        else:
            self.add(pr.SubtractMeanImage(mean))


class PreprocessMask(SequentialProcessor):
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessMask, self).__init__()
        self.add(pr.ResizeImage(shape))


class PreprocessBoxes(SequentialProcessor):
    def __init__(self, num_classes, prior_boxes, variances=[.1, .2]):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes),)
        # self.add(NormalizeBoxes())
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))
        # self.add(MatchRPNBoxes(config))


class AugmentBoxes(SequentialProcessor):
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToImageBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class DetectionPipeline(SequentialProcessor):
    def __init__(self, config, prior_boxes, num_classes=21,
                 size=640, mean=pr.BGR_IMAGENET_MEAN, split='TRAIN'):
        super(DetectionPipeline, self).__init__()

        self.augment_image = AugmentImage()
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        self.augment_boxes = AugmentBoxes()
        self.preprocess_boxes = PreprocessBoxes(num_classes, prior_boxes)

        self.preprocess_mask = PreprocessMask((size, size))
        self.add(pr.UnpackDictionary(['image', 'boxes', 'masks']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        self.add(pr.ControlMap(pr.LoadMask(), [2], [2]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.preprocess_boxes, [0, 1], [1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.ControlMap(self.preprocess_mask, [2], [2]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]},
             2: {'masks': [size, size, (num_classes - 1) * 3]}}))


class DataSequencer(SequenceExtra):
    def __init__(self, processor, batch_size, data, as_list=False):
        self.data = data
        super(DataSequencer, self).__init__(processor, batch_size,
                                            as_list)

    def process_batch(self, inputs, labels, batch_index):
        unprocessed_batch = self._get_unprocessed_batch(self.data, batch_index)

        for sample_arg, unprocessed_sample in enumerate(unprocessed_batch):
            sample = self.pipeline(unprocessed_sample.copy())
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)
        return inputs, labels
