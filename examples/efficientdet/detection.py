import numpy as np
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor, Box2D
from paz.pipelines.detection import DetectSingleShot
from paz.backend.image import resize_image
from paz.backend.image.draw import draw_rectangle
from boxes import nms_per_class
from draw import (compute_text_bounds, draw_opaque_box, make_box_transparent,
                  put_text)
from efficientdet import (EFFICIENTDETD0, EFFICIENTDETD1, EFFICIENTDETD2,
                          EFFICIENTDETD3, EFFICIENTDETD4, EFFICIENTDETD5,
                          EFFICIENTDETD6, EFFICIENTDETD7)

B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV = 57.3, 57.1, 58.4
RGB_IMAGENET_STDEV = (R_IMAGENET_STDEV, G_IMAGENET_STDEV, B_IMAGENET_STDEV)


class DetectSingleShotEfficientDet(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating class names.
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        mean: List of three elements indicating per channel mean.
        variances: List of floats indicating variances to be encoded
            for bounding boxes.
        draw: Bool. If ``True`` prediction are drawn on the
            returned image.

    # Properties
        model: Keras model.
        class_names: List.
        score_thresh: Float.
        nms_thresh: Float.
        variances: List.
        draw: Bool.
        model.prior_boxes: Array.

    # Methods
        call()
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 mean=pr.RGB_IMAGENET_MEAN, variances=[1.0, 1.0, 1.0, 1.0],
                 draw=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw
        self.model.prior_boxes = model.prior_boxes * model.input_shape[1]

        super(DetectSingleShotEfficientDet, self).__init__()
        preprocessing = SequentialProcessor([
            pr.CastImage(float),
            pr.SubtractMeanImage(mean=mean),
            DivideStandardDeviationImage(
                standard_deviation=RGB_IMAGENET_STDEV),
            ScaledResize(image_size=self.model.input_shape[1])])
        self.preprocessing = preprocessing

        self.draw_boxes2D = DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        preprocessed_image, image_scales = self.preprocessing(image)
        outputs = self.model(preprocessed_image)
        postprocessing = SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(self.model.prior_boxes, variances=self.variances),
            ScaleBox(image_scales),
            NonMaximumSuppressionPerClass(self.nms_thresh),
            FilterBoxes(self.class_names, self.score_thresh)])
        outputs = process_outputs(outputs)
        boxes2D = postprocessing(outputs)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class DivideStandardDeviationImage(Processor):
    """Divide channel-wise standard deviation to image.

    # Arguments
        standard_deviation: List of length 3, containing the
            channel-wise standard deviation.

    # Properties
        standard_deviation: List.

    # Methods
        call()
    """
    def __init__(self, standard_deviation):
        self.standard_deviation = standard_deviation
        super(DivideStandardDeviationImage, self).__init__()

    def call(self, image):
        return image / self.standard_deviation


class ScaledResize(Processor):
    """Resizes image by returning the scales to original image.

    # Arguments
        image_size: Int, desired size of the model input.

    # Properties
        image_size: Int.

    # Methods
        call()
    """
    def __init__(self, image_size):
        self.image_size = image_size
        super(ScaledResize, self).__init__()

    def call(self, image):
        """
        # Arguments
            image: Array, raw input image.
        """
        crop_offset_y = np.array(0)
        crop_offset_x = np.array(0)
        height = np.array(image.shape[0]).astype('float32')
        width = np.array(image.shape[1]).astype('float32')
        image_scale_y = np.array(self.image_size).astype('float32') / height
        image_scale_x = np.array(self.image_size).astype('float32') / width
        image_scale = np.minimum(image_scale_x, image_scale_y)
        scaled_height = (height * image_scale).astype('int32')
        scaled_width = (width * image_scale).astype('int32')
        scaled_image = resize_image(image, (scaled_width, scaled_height))
        scaled_image = scaled_image[
                       crop_offset_y: crop_offset_y + self.image_size,
                       crop_offset_x: crop_offset_x + self.image_size,
                       :]
        output_images = np.zeros((self.image_size,
                                  self.image_size,
                                  image.shape[2]))
        output_images[:scaled_image.shape[0],
                      :scaled_image.shape[1],
                      :scaled_image.shape[2]] = scaled_image
        image_scale = 1 / image_scale
        output_images = output_images[np.newaxis]
        return output_images, image_scale


class ScaleBox(Processor):
    """Scale box coordinates of the prediction.

    # Arguments
        scales: Array of shape `()`, value to scale boxes.

    # Properties
        scales: Int.

    # Methods
        call()
    """
    def __init__(self, scales):
        super(ScaleBox, self).__init__()
        self.scales = scales

    def call(self, boxes):
        boxes = scale_box(boxes, self.scales)
        return boxes


def scale_box(predictions, image_scales=None):
    """
    # Arguments
        predictions: Array of shape `(num_boxes, num_classes+N)`
            model predictions.
        image_scales: Array of shape `()`, scale value of boxes.

    # Returns
        predictions: Array of shape `(num_boxes, num_classes+N)`
            model predictions.
    """

    if image_scales is not None:
        boxes = predictions[:, :4]
        scales = image_scales[np.newaxis][np.newaxis]
        boxes = boxes * scales
        predictions = np.concatenate([boxes, predictions[:, 4:]], 1)
    return predictions


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
        names = get_class_names('COCO')
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
        names = get_class_names('COCO')
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
        names = get_class_names('COCO')
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
        names = get_class_names('COCO')
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
        names = get_class_names('COCO')
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
        names = get_class_names('COCO')
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
        names = get_class_names('COCO')
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


class DrawBoxes2D(pr.DrawBoxes2D):
    """Draws bounding boxes from Boxes2D messages.
    # Arguments
        class_names: List, class names.
        colors: List, color values.
        weighted: Bool, whether to weight bounding box color.
        scale: Float. Scale of text drawn.
        with_score: Bool, denoting if confidence be shown.
    # Methods
        compute_box_color()
        compute_text()
        get_text_box_parameters()
        call()
    """
    def __init__(
            self, class_names=None, colors=None,
            weighted=False, scale=0.7, with_score=True):
        super().__init__(
            class_names, colors, weighted, scale, with_score)

    def compute_box_color(self, box2D):
        class_name = box2D.class_name
        color = self.class_to_color[class_name]
        if self.weighted:
            color = [int(channel * box2D.score) for channel in color]
        return color

    def compute_text(self, box2D):
        class_name = box2D.class_name
        text = '{}'.format(class_name)
        if self.with_score:
            text = '{} :{}%'.format(class_name, round(box2D.score * 100))
        return text

    def get_text_box_parameters(self):
        thickness = 1
        offset_x = 2
        offset_y = 17
        color = (0, 0, 0)
        text_parameters = [thickness, offset_x, offset_y, color]
        box_start_offset = 2
        box_end_offset = 5
        box_color = (255, 174, 66)
        text_box_parameters = [box_start_offset, box_end_offset, box_color]
        return [text_box_parameters, text_parameters]

    def call(self, image, boxes2D):
        raw_image = image.copy()
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates.astype(np.int)
            color = self.compute_box_color(box2D)
            draw_opaque_box(image, (x_min, y_min), (x_max, y_max), color)
        image = make_box_transparent(raw_image, image)
        text_box_parameters, text_parameters = self.get_text_box_parameters()
        offset_start, offset_end, text_box_color = text_box_parameters
        text_thickness, offset_x, offset_y, text_color = text_parameters
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates.astype(np.int)
            color = self.compute_box_color(box2D)
            draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            text = self.compute_text(box2D)
            text_size = compute_text_bounds(text, self.scale, text_thickness)
            (text_W, text_H), _ = text_size
            args = (image, (x_min + offset_start, y_min + offset_start),
                    (x_min + text_W + offset_end, y_min + text_H + offset_end),
                    text_box_color)
            draw_opaque_box(*args)
            args = (image, text, (x_min + offset_x, y_min + offset_y),
                    self.scale, text_color, text_thickness)
            put_text(*args)
        return image


#################### Modifications to paz modules #######################
class ToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages.

    # Arguments
        class_names: List of class names ordered with respect to the class
            indices from the dataset ``boxes``.
    """
    def __init__(self, class_names=None, one_hot_encoded=False):
        if class_names is not None:
            self.arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        super(ToBoxes2D, self).__init__()

    def call(self, boxes):
        numpy_boxes2D, boxes2D = boxes, []
        for numpy_box2D in numpy_boxes2D:
            if self.one_hot_encoded:
                class_name = self.arg_to_class[np.argmax(numpy_box2D[4:])]
            elif numpy_box2D.shape[-1] == 5:
                class_name = self.arg_to_class[numpy_box2D[-1]]
            elif numpy_box2D.shape[-1] == 4:
                class_name = None
            boxes2D.append(Box2D(numpy_box2D[:4], 1.0, class_name))
        return boxes2D


class NonMaximumSuppressionPerClass(Processor):
    """Applies non maximum suppression per class.

    # Arguments
        nms_thresh: Float between [0, 1].
        conf_thresh: Float between [0, 1].
    """
    def __init__(self, nms_thresh=.45, conf_thresh=0.01):
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        super(NonMaximumSuppressionPerClass, self).__init__()

    def call(self, boxes):
        boxes = nms_per_class(boxes, self.nms_thresh, self.conf_thresh)
        return boxes


class FilterBoxes(Processor):
    """Filters boxes outputted from function ``detect`` as ``Box2D`` messages.

    # Arguments
        class_names: List of class names.
        conf_thresh: Float between [0, 1].
    """
    def __init__(self, class_names, conf_thresh=0.5):
        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.arg_to_class = dict(zip(
            list(range(len(self.class_names))), self.class_names))
        super(FilterBoxes, self).__init__()

    def call(self, boxes):
        num_classes = boxes.shape[0]
        boxes2D = []
        for class_arg in range(num_classes):
            class_detections = boxes[class_arg, :]
            confidence_mask = np.squeeze(
                class_detections[:, -1] >= self.conf_thresh)
            confident_class_detections = class_detections[confidence_mask]
            if len(confident_class_detections) == 0:
                continue
            class_name = self.arg_to_class[class_arg]
            for confident_class_detection in confident_class_detections:
                coordinates = confident_class_detection[:4]
                score = confident_class_detection[4]
                boxes2D.append(Box2D(coordinates, score, class_name))
        return boxes2D
