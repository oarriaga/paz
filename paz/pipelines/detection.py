from ..abstract import SequentialProcessor, Processor
from .. import processors as pr
from ..models import SSD512, SSD300
from ..datasets import get_class_names

from .image import AugmentImage, PreprocessImage


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.
    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToAbsoluteBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ''[num_boxes, 4]'' containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(pr.MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.
    # Arguments
        prior_boxes: Numpy array of shape ''[num_boxes, 4]'' containing
            prior/default bounding boxes.
        split: Flag from ''paz.processors.TRAIN'', ''paz.processors.VAL''
            or ''paz.processors.TEST''. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, split=pr.TRAIN, num_classes=21, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5, variances=[.1, .2]):
        super(AugmentDetection, self).__init__()

        self.augment_image = AugmentImage()
        self.augment_image.insert(0, pr.LoadImage())
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        self.add(pr.UnpackDictionary(['image', 'boxes']))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


class SingleShotPrediction(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.BGR_IMAGENET_MEAN):
        super(SingleShotPrediction, self).__init__()
        preprocessing = SequentialProcessor(
            [pr.ResizeImage(model.input_shape[1:3]),
             pr.ConvertColorSpace(pr.RGB2BGR),
             pr.SubtractMeanImage(mean),
             pr.CastImage(float),
             pr.ExpandDims(axis=0)])
        postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(model.prior_boxes, variances=[.1, .2]),
             pr.NonMaximumSuppressionPerClass(nms_thresh),
             pr.FilterBoxes(class_names, score_thresh)])
        self.predict = pr.Predict(model, preprocessing, postprocessing)

        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw = pr.DrawBoxes2D(class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


class SSD512COCO(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD512 trained on COCO.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD512()
        names = get_class_names('COCO')
        super(SSD512COCO, self).__init__(
            model, names, score_thresh, nms_thresh)


class SSD512YCBVideo(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD512 trained on YCBVideo.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD512(weights='YCBVideo')
        names = get_class_names('YCBVideo')
        super(SSD512YCBVideo, self).__init__(
            model, names, score_thresh, nms_thresh)


class SSD300VOC(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD300 trained on VOC.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD300()
        names = get_class_names('VOC')
        super(SSD300VOC, self).__init__(model, names, score_thresh, nms_thresh)


class SSD300FAT(SingleShotPrediction):
    def __init__(self, score_thresh=0.60, nms_thresh=0.45):
        """Single-shot inference pipeline with SSD300 trained on FAT.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        """
        model = SSD300(22, 'FAT', 'FAT')
        names = get_class_names('FAT')
        super(SSD300FAT, self).__init__(model, names, score_thresh, nms_thresh)
