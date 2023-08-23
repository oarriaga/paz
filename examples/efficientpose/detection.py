from efficientpose import EFFICIENTPOSEA
from paz.abstract import Processor, SequentialProcessor
import paz.processors as pr
from processors import ComputeResizingShape, PadImage, ComputeCameraParameter
import numpy as np
from paz.backend.boxes import change_box_coordinates


B_LINEMOD_MEAN, G_LINEMOD_MEAN, R_LINEMOD_MEAN = 103.53, 116.28, 123.675
RGB_LINEMOD_MEAN = (R_LINEMOD_MEAN, G_LINEMOD_MEAN, B_LINEMOD_MEAN)
B_LINEMOD_STDEV, G_LINEMOD_STDEV, R_LINEMOD_STDEV = 57.375, 57.12, 58.395
RGB_LINEMOD_STDEV = (R_LINEMOD_STDEV, G_LINEMOD_STDEV, B_LINEMOD_STDEV)

LINEMOD_CAMERA_MATRIX = np.array([
    [572.4114, 0., 325.2611],
    [0., 573.57043, 242.04899],
    [0., 0., 1.]],
    dtype=np.float32)


def get_class_names(dataset_name='LINEMOD'):
    if dataset_name in ['LINEMOD']:
        class_names = ['ape', 'can', 'cat', 'driller', 'duck',
                       'eggbox', 'glue', 'holepuncher']

    return class_names


class DetectAndEstimateSingleShot(Processor):
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
            self.preprocess = EfficientPosePreprocess(model)
        if postprocess is None:
            self.postprocess = EfficientPosePostprocess(
                model, class_names, score_thresh, nms_thresh)

        super(DetectAndEstimateSingleShot, self).__init__()
        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        preprocessed_data = self.preprocess(image)
        preprocessed_image, image_scale, camera_parameter = preprocessed_data
        outputs = self.model(preprocessed_image)
        detections, (rotations, translations) = outputs
        detections = change_box_coordinates(detections)
        boxes2D = self.postprocess(detections)
        # boxes2D = self.denormalize(image, boxes2D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


class EfficientPosePreprocess(Processor):
    """Preprocessing pipeline for SSD.

    # Arguments
        model: Keras model.
        mean: List, of three elements indicating the per channel mean.
        color_space: Int, specifying the color space to transform.
    """
    def __init__(self, model, mean=RGB_LINEMOD_MEAN,
                 standard_deviation=RGB_LINEMOD_STDEV,
                 camera_matrix=LINEMOD_CAMERA_MATRIX,
                 translation_scale_norm=1000.0):
        super(EfficientPosePreprocess, self).__init__()

        self.compute_resizing_shape = ComputeResizingShape(
            model.input_shape[1])
        self.preprocess = pr.SequentialProcessor([
            pr.SubtractMeanImage(mean),
            pr.DivideStandardDeviationImage(standard_deviation),
            PadImage(model.input_shape[1]),
            pr.CastImage(float),
            pr.ExpandDims(axis=0)])
        self.compute_camera_parameter = ComputeCameraParameter(
            camera_matrix, translation_scale_norm)

    def call(self, image):
        resizing_shape, image_scale = self.compute_resizing_shape(image)
        resize_image = pr.ResizeImage(resizing_shape)
        preprocessed_image = resize_image(image)
        preprocessed_image = self.preprocess(preprocessed_image)
        camera_parameter = self.compute_camera_parameter(image_scale)
        return preprocessed_image, image_scale, camera_parameter


class EfficientPosePostprocess(Processor):
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
                 variances=[1.0, 1.0, 1.0, 1.0], class_arg=0, box_method=0):
        super(EfficientPosePostprocess, self).__init__()

        self.postprocess = pr.SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(model.prior_boxes*512, variances),
            pr.NonMaximumSuppressionPerClass(nms_thresh),
            pr.MergeNMSBoxWithClass(),
            pr.FilterBoxes(class_names, score_thresh),
            pr.ToBoxes2D(class_names, box_method)
            ])

    def call(self, output):
        postprocessed = self.postprocess(output)
        return postprocessed


class EFFICIENTPOSEALINEMOD(DetectAndEstimateSingleShot):
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
        names = get_class_names('LINEMOD')
        model = EFFICIENTPOSEA(num_classes=len(names),
                               base_weights='COCO', head_weights='COCO')
        super(EFFICIENTPOSEALINEMOD, self).__init__(
            model, names, score_thresh, nms_thresh, draw=draw)
