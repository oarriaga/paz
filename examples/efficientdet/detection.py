from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
from processors import MatchBoxes
from utils import efficientdet_preprocess
from efficientdet_postprocess import process_outputs
from utils import get_class_name_efficientdet


class AugmentImage(SequentialProcessor):
    """Augments an RGB image by randomly changing contrast, brightness
        saturation and hue.
    """
    def __init__(self):
        super(AugmentImage, self).__init__()
        self.add(pr.RandomContrast())
        self.add(pr.RandomBrightness())
        self.add(pr.RandomSaturation(0.7))
        self.add(pr.RandomHue())


class PreprocessImage(SequentialProcessor):
    """Preprocess RGB image by resizing it to the given ``shape``. If a
    ``mean`` is given it is substracted from image and it not the image gets
    normalized.

    # Argumeqnts
        shape: List of two Ints.
        mean: List of three Ints indicating the per-channel mean to be
            subtracted.
    """
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        self.add(pr.SubtractMeanImage(pr.RGB_IMAGENET_MEAN))
        self.add(pr.DivideStandardDeviationImage(pr.RGB_IMAGENET_STDEV))
        # if mean is None:
        #     self.add(pr.NormalizeImage())
        # else:
        #     print('Normal')
        #     self.add(pr.SubtractMeanImage(mean))


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.

    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToImageBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        # RandomSampleCrop was commented out
        self.add(pr.RandomSampleCrop())
        self.add(pr.RandomFlipBoxesLeftRight())
        self.add(pr.ToNormalizedBoxCoordinates())


class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(MatchBoxes(prior_boxes, IOU),)
        self.add(pr.EncodeBoxes(prior_boxes, variances))
        self.add(pr.BoxClassToOneHotVector(num_classes))


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.

    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, split=pr.TRAIN, num_classes=21, size=300,
                 mean=pr.BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[1, 1, 1, 1]):
        super(AugmentDetection, self).__init__()
        # image processors
        self.augment_image = AugmentImage()
        self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)

        # box processors
        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_image, [0], [0]))
            self.add(pr.ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0]))
        self.add(pr.ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))


class DetectSingleShot_EfficientDet(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating the per channel mean.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.image_size = model.input_shape[1]
        super(DetectSingleShot_EfficientDet, self).__init__()
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        image, image_scales = efficientdet_preprocess(image, self.image_size)
        outputs = self.model(image)
        outputs = process_outputs(outputs)
        postprocessing = SequentialProcessor(
            [pr.Squeeze(axis=None),
             pr.DecodeBoxes(self.model.prior_boxes, variances=[1, 1, 1, 1]),
             pr.ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
             pr.FilterBoxes(get_class_name_efficientdet('VOC'), 0.4)])
        outputs = postprocessing(outputs)
        draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('VOC'))
        image = draw_boxes2D(image.astype('uint8'), outputs)
        return self.wrap(image, outputs)
