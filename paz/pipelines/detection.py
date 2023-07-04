import numpy as np

from .. import processors as pr
from ..abstract import SequentialProcessor, Processor
from ..models import (
    SSD512, SSD300, HaarCascadeDetector, EFFICIENTDETD0, EFFICIENTDETD1,
    EFFICIENTDETD2, EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
    EFFICIENTDETD6, EFFICIENTDETD7)
from ..datasets import get_class_names

from .image import AugmentImage, PreprocessImage
from .classification import MiniXceptionFER
from .keypoints import FaceKeypointNet2D32, DetectMinimalHand
from .keypoints import MinimalHandPoseEstimation
from ..backend.boxes import change_box_coordinates


class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.

    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=pr.BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        self.add(pr.ToImageBoxCoordinates())
        self.add(pr.Expand(mean=mean))
        self.add(pr.RandomSampleCrop(1.0))
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
        self.add(pr.MatchBoxes(prior_boxes, IOU),)
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
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentDetection, self).__init__()
        # image processors
        self.augment_image = AugmentImage()
        # self.augment_image.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.preprocess_image = PreprocessImage((size, size), mean)
        self.preprocess_image.insert(0, pr.ConvertColorSpace(pr.RGB2BGR))

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


class PostprocessBoxes2D(SequentialProcessor):
    """Filters, squares and offsets 2D bounding boxes

    # Arguments
        valid_names: List of strings containing class names to keep.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
    """
    def __init__(self, offsets, valid_names=None):
        super(PostprocessBoxes2D, self).__init__()
        if valid_names is not None:
            self.add(pr.FilterClassBoxes2D(valid_names))
        self.add(pr.SquareBoxes2D())
        self.add(pr.OffsetBoxes2D(offsets))


class DetectSingleShot(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        preprocess: Callable, pre-processing pipeline.
        postprocess: Callable, post-processing pipeline.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        variances: List, of floats.
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 preprocess=None, postprocess=None,
                 variances=[0.1, 0.1, 0.2, 0.2], draw=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw
        if preprocess is None:
            preprocess = SSDPreprocess(model)
        if postprocess is None:
            postprocess = SSDPostprocess(
                model, class_names, score_thresh, nms_thresh)

        super(DetectSingleShot, self).__init__()
        self.predict = pr.Predict(self.model, preprocess, postprocess)
        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class SSDPreprocess(SequentialProcessor):
    """Preprocessing pipeline for SSD.

    # Arguments
        model: Keras model.
        mean: List, of three elements indicating the per channel mean.
        color_space: Int, specifying the color space to transform.
    """
    def __init__(
            self, model, mean=pr.BGR_IMAGENET_MEAN, color_space=pr.RGB2BGR):
        super(SSDPreprocess, self).__init__()
        self.add(pr.ResizeImage(model.input_shape[1:3]))
        self.add(pr.ConvertColorSpace(color_space))
        self.add(pr.SubtractMeanImage(mean))
        self.add(pr.CastImage(float))
        self.add(pr.ExpandDims(axis=0))


class SSDPostprocess(SequentialProcessor):
    """Postprocessing pipeline for SSD.

    # Arguments
        model: Keras model.
        class_names: List, of strings indicating the class names.
        score_thresh: Float, between [0, 1]
        nms_thresh: Float, between [0, 1].
        variances: List, of floats.
        class_arg: Int, index of class to be removed.
        box_method: Int, type of boxes to boxes2D conversion method.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 variances=[0.1, 0.1, 0.2, 0.2], class_arg=0, box_method=0):
        super(SSDPostprocess, self).__init__()
        self.add(pr.Squeeze(axis=None))
        self.add(pr.DecodeBoxes(model.prior_boxes, variances))
        self.add(pr.RemoveClass(class_names, class_arg, renormalize=False))
        self.add(pr.NonMaximumSuppressionPerClass(nms_thresh))
        self.add(pr.MergeNMSBoxWithClass())
        self.add(pr.FilterBoxes(class_names, score_thresh))
        self.add(pr.ToBoxes2D(class_names, box_method))


class DetectSingleShotEfficientDet(Processor):
    """Single-shot object detection prediction for EfficientDet models.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating class names.
        preprocess: Callable, preprocessing pipeline.
        postprocess: Callable, postprocessing pipeline.
        draw: Bool. If ``True`` prediction are drawn on the
            returned image.

    # Properties
        model: Keras model.
        draw: Bool.
        preprocess: Callable.
        postprocess: Callable.
        draw_boxes2D: Callable.
        wrap: Callable.

    # Methods
        call()
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 preprocess=None, postprocess=None, draw=True):
        self.model = model
        self.draw = draw
        self.draw_boxes2D = pr.DrawBoxes2D(class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])
        if preprocess is None:
            self.preprocess = EfficientDetPreprocess(model)
        if postprocess is None:
            self.postprocess = EfficientDetPostprocess(
                model, class_names, score_thresh, nms_thresh)
        super(DetectSingleShotEfficientDet, self).__init__()

    def call(self, image):
        preprocessed_image, image_scales = self.preprocess(image)
        outputs = self.model(preprocessed_image)
        outputs = change_box_coordinates(outputs)
        boxes2D = self.postprocess(outputs, image_scales)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class EfficientDetPreprocess(SequentialProcessor):
    """Preprocessing pipeline for EfficientDet.

    # Arguments
        model: Keras model.
        mean: Tuple, containing mean per channel on ImageNet.
        standard_deviation: Tuple, containing standard deviations
            per channel on ImageNet.
    """
    def __init__(self, model, mean=pr.RGB_IMAGENET_MEAN,
                 standard_deviation=pr.RGB_IMAGENET_STDEV):
        super(EfficientDetPreprocess, self).__init__()
        self.add(pr.CastImage(float))
        self.add(pr.SubtractMeanImage(mean=mean))
        self.add(pr.DivideStandardDeviationImage(standard_deviation))
        self.add(pr.ScaledResize(image_size=model.input_shape[1]))


class EfficientDetPostprocess(Processor):
    """Postprocessing pipeline for EfficientDet.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating class names.
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        variances: List of float values.
        class_arg: Int, index of the class to be removed.
        renormalize: Bool, if true scores are renormalized.
        method: Int, method to convert boxes to ``Boxes2D``.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 variances=[1.0, 1.0, 1.0, 1.0], class_arg=None):
        super(EfficientDetPostprocess, self).__init__()
        model.prior_boxes = model.prior_boxes * model.input_shape[1]
        self.postprocess = pr.SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(model.prior_boxes, variances),
            pr.RemoveClass(class_names, class_arg)])
        self.scale = pr.ScaleBox()
        self.nms_per_class = pr.NonMaximumSuppressionPerClass(nms_thresh)
        self.merge_box_and_class = pr.MergeNMSBoxWithClass()
        self.filter_boxes = pr.FilterBoxes(class_names, score_thresh)
        self.to_boxes2D = pr.ToBoxes2D(class_names)
        self.round_boxes = pr.RoundBoxes2D()

    def call(self, output, image_scale):
        box_data = self.postprocess(output)
        box_data = self.scale(box_data, image_scale)
        box_data, class_labels = self.nms_per_class(box_data)
        box_data = self.merge_box_and_class(box_data, class_labels)
        box_data = self.filter_boxes(box_data)
        boxes2D = self.to_boxes2D(box_data)
        boxes2D = self.round_boxes(boxes2D)
        return boxes2D


class SSD512COCO(DetectSingleShot):
    """Single-shot inference pipeline with SSD512 trained on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Example
        ``` python
        from paz.pipelines import SSD512COCO

        detect = SSD512COCO()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
     # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    # Reference
        - [SSD: Single Shot MultiBox
            Detector](https://arxiv.org/abs/1512.02325)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        model = SSD512()
        names = get_class_names('COCO')
        super(SSD512COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class SSD512YCBVideo(DetectSingleShot):
    """Single-shot inference pipeline with SSD512 trained on YCBVideo.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Example
        ``` python
        from paz.pipelines import SSD512YCBVideo

        detect = SSD512YCBVideo()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```

    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('YCBVideo')
        model = SSD512(head_weights='YCBVideo', num_classes=len(names))
        super(SSD512YCBVideo, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class SSD300VOC(DetectSingleShot):
    """Single-shot inference pipeline with SSD300 trained on VOC.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Example
        ``` python
        from paz.pipelines import SSD300VOC

        detect = SSD300VOC()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```

    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    # Reference
        - [SSD: Single Shot MultiBox
            Detector](https://arxiv.org/abs/1512.02325)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        model = SSD300()
        names = get_class_names('VOC')
        super(SSD300VOC, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class SSD300FAT(DetectSingleShot):
    """Single-shot inference pipeline with SSD300 trained on FAT.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Example
        ``` python
        from paz.pipelines import SSD300FAT

        detect = SSD300FAT()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        model = SSD300(22, 'FAT', 'FAT')
        names = get_class_names('FAT')
        super(SSD300FAT, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class DetectHaarCascade(Processor):
    """HaarCascade prediction pipeline/function from RGB-image.

    # Arguments
        detector: An instantiated ``HaarCascadeDetector`` model.
        offsets: List of two elements. Each element must be between [0, 1].
        class_names: List of strings.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Returns
        A function for predicting bounding box detections.
    """
    def __init__(self, detector, class_names=None, colors=None, draw=True):
        super(DetectHaarCascade, self).__init__()
        self.detector = detector
        self.class_names = class_names
        self.colors = colors
        self.draw = draw
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        postprocess = SequentialProcessor()
        postprocess.add(pr.ToBoxes2D(self.class_names, box_method=2))
        self.predict = pr.Predict(self.detector, RGB2GRAY, postprocess)
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names, self.colors)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class HaarCascadeFrontalFace(DetectHaarCascade):
    """HaarCascade pipeline for detecting frontal faces

    # Arguments
        class_name: String indicating the class name.
        color: List indicating the RGB color e.g. ``[0, 255, 0]``.
        draw: Boolean. If ``False`` the bounding boxes are not drawn.

    # Example
        ``` python
        from paz.pipelines import HaarCascadeFrontalFace

        detect = HaarCascadeFrontalFace()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    """
    def __init__(self, class_name='Face', color=[0, 255, 0], draw=True):
        self.model = HaarCascadeDetector('frontalface_default', class_arg=0)
        super(HaarCascadeFrontalFace, self).__init__(
            self.model, [class_name], [color], draw)


EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]


class DetectMiniXceptionFER(Processor):
    """Emotion classification and detection pipeline.

    # Returns
        Dictionary with ``image`` and ``boxes2D``.

    # Example
        ``` python
        from paz.pipelines import DetectMiniXceptionFER

        detect = DetectMiniXceptionFER()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    def __init__(self, offsets=[0, 0], colors=EMOTION_COLORS):
        super(DetectMiniXceptionFER, self).__init__()
        self.offsets = offsets
        self.colors = colors

        # detection
        self.detect = HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()

        # classification
        self.classify = MiniXceptionFER()

        # drawing and wrapping
        self.class_names = self.classify.class_names
        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image.copy())['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            predictions = self.classify(cropped_image)
            box2D.class_name = predictions['class_name']
            box2D.score = np.amax(predictions['scores'])
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)


class DetectKeypoints2D(Processor):
    def __init__(self, detect, estimate_keypoints, offsets=[0, 0], radius=3):
        """General detection and keypoint estimator pipeline.

        # Arguments
            detect: Function for detecting objects. The output should be a
                dictionary with key ``Boxes2D`` containing a list
                of ``Boxes2D`` messages.
            estimate_keypoints: Function for estimating keypoints. The output
                should be a dictionary with key ``keypoints`` containing
                a numpy array of keypoints.
            offsets: List of two elements. Each element must be between [0, 1].
            radius: Int indicating the radius of the keypoints to be drawn.
        """
        super(DetectKeypoints2D, self).__init__()
        self.detect = detect
        self.estimate_keypoints = estimate_keypoints
        self.num_keypoints = estimate_keypoints.num_keypoints
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, radius, False)
        self.draw_boxes = pr.DrawBoxes2D(detect.class_names, detect.colors)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'keypoints'])

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        keypoints2D = []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)['keypoints']
            keypoints = self.change_coordinates(keypoints, box2D)
            keypoints2D.append(keypoints)
            image = self.draw(image, keypoints)
        image = self.draw_boxes(image, boxes2D)
        return self.wrap(image, boxes2D, keypoints2D)


class DetectFaceKeypointNet2D32(DetectKeypoints2D):
    """Frontal face detection pipeline with facial keypoint estimation.

    # Arguments
        offsets: List of two elements. Each element must be between [0, 1].
        radius: Int indicating the radius of the keypoints to be drawn.

    # Example
        ``` python
        from paz.pipelines import DetectFaceKeypointNet2D32

        detect = DetectFaceKeypointNet2D32()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    """
    def __init__(self, offsets=[0, 0], radius=3):
        detect = HaarCascadeFrontalFace(draw=False)
        estimate_keypoints = FaceKeypointNet2D32(draw=False)
        super(DetectFaceKeypointNet2D32, self).__init__(
            detect, estimate_keypoints, offsets, radius)


class SSD512HandDetection(DetectSingleShot):
    """Minimal hand detection with SSD512Custom trained on OPenImageV6.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Example
        ``` python
        from paz.pipelines import SSD512HandDetection

        detect = SSD512HandDetection()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
     # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    # Reference
        - [SSD: Single Shot MultiBox
            Detector](https://arxiv.org/abs/1512.02325)
    """
    def __init__(self, score_thresh=0.40, nms_thresh=0.45, draw=True):
        class_names = ['background', 'hand']
        num_classes = len(class_names)
        model = SSD512(num_classes, base_weights='OIV6Hand',
                       head_weights='OIV6Hand')
        super(SSD512HandDetection, self).__init__(
            model, class_names, score_thresh, nms_thresh, draw=draw)


class SSD512MinimalHandPose(DetectMinimalHand):
    """Hand detection and minimal hand pose estimation pipeline.

    # Arguments
        right_hand: Boolean. True for right hand inference.
        offsets: List of two elements. Each element must be between [0, 1].

    # Example
        ``` python
        from paz.pipelines import SSD512MinimalHandPose

        detect = SSD512MinimalHandPose()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```

    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image``,  ``boxes2D``,
        ``Keypoints2D``, ``Keypoints3D``.
        The corresponding values of these keys contain the image with the drawn
        inferences.
    """
    def __init__(self, right_hand=False, offsets=[0.25, 0.25]):
        detector = SSD512HandDetection()
        keypoint_estimator = MinimalHandPoseEstimation(right_hand)
        super(SSD512MinimalHandPose, self).__init__(
            detector, keypoint_estimator, offsets)


class EFFICIENTDETD0COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD0 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD0(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD0COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD1COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD1 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD1(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD1COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD2COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD2 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD2(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD2COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD3COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD3 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD3(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD3COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD4COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD4 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD4(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD4COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD5COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD5 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD5(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD5COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD6COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD6 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD6(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD6COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD7COCO(DetectSingleShotEfficientDet):
    """Single-shot inference pipeline with EFFICIENTDETD7 trained
    on COCO.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO_EFFICIENTDET')
        model = EFFICIENTDETD7(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTDETD7COCO, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)


class EFFICIENTDETD0VOC(DetectSingleShot):
    """Single-shot inference pipeline with EFFICIENTDETD0 trained
    on VOC.

    # Arguments
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.

    # References
        [Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('VOC')
        model = EFFICIENTDETD0(num_classes=len(names),
                               base_weights='VOC', head_weights='VOC')
        super(EFFICIENTDETD0VOC, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)
