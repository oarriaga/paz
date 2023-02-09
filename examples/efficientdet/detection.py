import numpy as np
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
from paz.pipelines.detection import DetectSingleShot
from efficientdet import (EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2,
                          EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
                          EFFICIENTDETD6, EFFICIENTDETD7)
from processors import (DivideStandardDeviationImage, ScaledResize, ScaleBox,
                        NonMaximumSuppressionPerClass, FilterBoxes,
                        ToBoxes2D, RemoveClass)

B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV = 57.3, 57.1, 58.4
RGB_IMAGENET_STDEV = (R_IMAGENET_STDEV, G_IMAGENET_STDEV, B_IMAGENET_STDEV)


class DetectSingleShotEfficientDet(Processor):
    """Single-shot object detection prediction.

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
    def __init__(self, model, class_names, preprocess, postprocess, draw=True):
        self.model = model
        self.draw = draw
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.draw_boxes2D = pr.DrawBoxes2D(class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])
        super(DetectSingleShotEfficientDet, self).__init__()

    def call(self, image):
        preprocessed_image, image_scales = self.preprocess(image)
        outputs = self.model(preprocessed_image)
        outputs = process_outputs(outputs)
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
                 standard_deviation=RGB_IMAGENET_STDEV):
        super(EfficientDetPreprocess, self).__init__()
        self.add(pr.CastImage(float))
        self.add(pr.SubtractMeanImage(mean=mean))
        self.add(DivideStandardDeviationImage(standard_deviation))
        self.add(ScaledResize(image_size=model.input_shape[1]))


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
            RemoveClass(class_names, class_arg)])
        self.scale = ScaleBox()
        self.nms_per_class = NonMaximumSuppressionPerClass(nms_thresh)
        self.filter_boxes = FilterBoxes(score_thresh)
        self.to_boxes2D = ToBoxes2D(class_names)

    def call(self, output, image_scales):
        box_data = self.postprocess(output)
        box_data = self.scale(box_data, image_scales)
        box_data = self.nms_per_class(box_data)
        box_data = self.filter_boxes(box_data)
        boxes2D = self.to_boxes2D(box_data)
        return boxes2D


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD0(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD0COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD1(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD1COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD2(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD2COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
    def __init__(
            self, score_thresh=0.60, nms_thresh=0.45, draw=True):
        names = get_class_names('COCO')
        model = EFFICIENTDETD3(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD3COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD4(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD4COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD5(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD5COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD6(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD6COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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
        names = get_class_names('COCO')
        model = EFFICIENTDETD7(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        preprocess = EfficientDetPreprocess(model)
        postprocess = EfficientDetPostprocess(
            model, names, score_thresh, nms_thresh)
        super(EFFICIENTDETD7COCO, self).__init__(
            model, names, preprocess, postprocess, draw=draw)


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


def process_outputs(outputs):
    """Merges all feature levels into single tensor and combines
    box offsets and class scores.

    # Arguments
        outputs: Tensor, model output.

    # Returns
        outputs: Array, Processed outputs by merging the features
            at all levels. Each row corresponds to box coordinate
            offsets and sigmoid of the class logits.
    """
    outputs = outputs[0]
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = np.hsplit(boxes, 4)
    boxes = np.concatenate([s2, s1, s4, s3], axis=1)
    boxes = boxes[np.newaxis]
    classes = classes[np.newaxis]
    outputs = np.concatenate([boxes, classes], axis=2)
    return outputs


def get_class_names(dataset_name):
    if dataset_name == 'COCO':
        return ['person', 'bicycle', 'car', 'motorcycle',
                'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '0', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', '0', 'backpack', 'umbrella', '0',
                '0', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', '0', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', '0', 'dining table', '0', '0',
                'toilet', '0', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '0', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    elif dataset_name == 'VOC':
        return ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
