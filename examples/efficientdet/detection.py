from paz import processors as pr
from paz.abstract import Processor, SequentialProcessor
from paz.pipelines import DetectSingleShot
from paz.pipelines.detection import AugmentBoxes, PreprocessBoxes
from paz.pipelines.image import AugmentImage

import necessary_imports as ni
from draw import (add_box_border, draw_opaque_box, get_text_size,
                  make_box_transparent, put_text)
from efficientdet_postprocess import process_outputs
from utils import efficientdet_preprocess, get_class_name_efficientdet


class PreprocessImage(SequentialProcessor):
    """Preprocess RGB image.

    # Arguments
        shape: List, being desired image width and height.
        mean: List, being per-channel mean.
    """
    def __init__(self, shape, mean=pr.BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        self.add(pr.ResizeImage(shape))
        self.add(pr.CastImage(float))
        self.add(pr.SubtractMeanImage(pr.RGB_IMAGENET_MEAN))
        self.add(ni.DivideStandardDeviationImage(ni.RGB_IMAGENET_STDEV))


class AugmentDetection(SequentialProcessor):
    """Augment boxes and images.

    # Arguments
        prior_boxes: Numpy array, holding prior boxes.
        split: Flag denoting additional transformation.
        num_classes: Int, being number of classes.
        size: Int. being image size.
        mean: List, being per-channel mean.
        IOU: Float. Intersection over union.
        variances: List, bounding boxes encoding variances.
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
        class_names: List, indicating class names.
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        mean: List, being per-channel mean.
        draw: Bool, denoting if detections be drawn on image.
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
             ni.ScaleBox(image_scales), pr.NonMaximumSuppressionPerClass(0.4),
             pr.FilterBoxes(get_class_name_efficientdet('VOC'), 0.4)])
        outputs = postprocessing(outputs)
        draw_boxes2D = pr.DrawBoxes2D(get_class_name_efficientdet('VOC'))
        image = draw_boxes2D(image.astype('uint8'), outputs)
        return self.wrap(image, outputs)


class DetectSingleShot(DetectSingleShot):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List, indicating class names.
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        mean: List, being per-channel mean.
        variances: List, bounding boxes encoding variances.
        draw: Bool, denoting if detections be shown on image.
    """
    def __init__(
            self, model, class_names, score_thresh, nms_thresh,
            mean=pr.BGR_IMAGENET_MEAN, variances=[0.1, 0.1, 0.2, 0.2],
            draw=True):
        super().__init__(
            model, class_names, score_thresh, nms_thresh,
            mean, variances, draw)
        self.draw_boxes2D = DrawBoxes2D(class_names)


class DrawBoxes2D(pr.DrawBoxes2D):
    """Draws bounding boxes from Boxes2D messages.

    # Arguments
        class_names: List, indicating class names.
        colors: List holding color values.
        weighted: Bool, denoting bounding box color to be weighted.
        scale: Float. Scale of text drawn.
        with_score: Bool, denoting if confidence be shown.
    """
    def __init__(
            self, class_names=None, colors=None,
            weighted=False, scale=0.7, with_score=True):
        super().__init__(
            class_names, colors, weighted, scale, with_score)

    def compute_prediction_parameters(self, box2D):
        x_min, y_min, x_max, y_max = box2D.coordinates
        class_name = box2D.class_name
        color = self.class_to_color[class_name]
        if self.weighted:
            color = [int(channel * box2D.score) for channel in color]
        if self.with_score:
            text = '{} :{}%'.format(class_name, round(box2D.score*100))
        if not self.with_score:
            text = '{}'.format(class_name)
        return x_min, y_min, x_max, y_max, color, text

    def call(self, image, boxes2D):
        raw_image = image.copy()
        for box2D in boxes2D:
            prediction_parameters = self.compute_prediction_parameters(box2D)
            x_min, y_min, x_max, y_max, color, text = prediction_parameters
            draw_opaque_box(image, (x_min, y_min), (x_max, y_max), color)
        image = make_box_transparent(raw_image, image)
        for box2D in boxes2D:
            prediction_parameters = self.compute_prediction_parameters(box2D)
            x_min, y_min, x_max, y_max, color, text = prediction_parameters
            add_box_border(image, (x_min, y_min), (x_max, y_max), color, 2)
            text_size = get_text_size(text, self.scale, 1)
            (text_W, text_H), _ = text_size
            args = (image, (x_min + 2, y_min + 2),
                    (x_min + text_W + 5, y_min + text_H + 5), (255, 174, 66))
            draw_opaque_box(*args)
            args = (image, text, (x_min + 2, y_min + 17), self.scale,
                    (0, 0, 0), 1)
            put_text(*args)
        return image
