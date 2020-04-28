from .abstract import Processor, SequentialProcessor
from . import processors as pr


class AugmentImage(SequentialProcessor):
    def __init__(self):
        super(AugmentImage, self).__init__()
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())
        self.add(pr.RandomSaturation())
        self.add(pr.RandomHue())


class AugmentBoxes(SequentialProcessor):
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        self.mean = mean
        self.add(pr.Expand(mean=self.mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.HorizontalFlip())


class PreprocessImage(SequentialProcessor):
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        self.shape = shape
        self.mean = mean
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImageToFloat())
        if self.mean is None:
            self.add(pr.NormalizeImage())
        self.add(pr.SubtractMeanImage(self.mean))


class DetectionAugmentation(Processor):
    def __init__(self, prior_boxes, num_classes, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5, variances=[.1, .2]):
        super(DetectionAugmentation, self).__init__()
        self.prior_boxes = prior_boxes
        self.num_classes = num_classes
        self.size = size
        self.mean = mean
        self.IOU = IOU
        self.variances = variances

        self.load_image = pr.LoadImage()
        self.to_absolute_coordinates = pr.ToAbsoluteCoordinates()
        self.to_percent_coordinates = pr.ToPercentCoordinates()
        self.match_boxes = pr.MatchBoxes(self.prior_boxes, self.iou)
        self.encode_boxes = pr.EncodeBoxes(self.prior_boxes, self.variances)
        self.to_one_hot_vector = pr.BoxClassToOneHotVector(num_classes)

    def call(self, image_path, boxes):
        image = self.load_image(image_path)
        image = self.augment_image(image)
        boxes = self.to_absolute_coordinates(boxes)
        image, boxes = self.box_augmentation(image, boxes)
        boxes = self.to_percent_coordinates(boxes)
        image = self.preprocess_image(image)
        boxes = self.match_boxes(boxes)
        boxes = self.encode_boxes(boxes)
        boxes = self.to_one_hot_vector(boxes)
        return image, boxes
