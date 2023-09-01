import numpy as np
from efficientpose import EFFICIENTPOSEA
from paz.abstract import Processor
import paz.processors as pr
from processors import (ComputeResizingShape, PadImage, ComputeCameraParameter,
                        RegressTranslation, ComputeTxTy,
                        ComputeSelectedIndices, ToPose6D, DrawPoses6D)
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

LINEMOD_OBJECT_SIZES = {
    "ape":         np.array([75.86860000, 77.59920000, 91.76900000]),
    "benchvise":   np.array([215.67000000, 121.85570000, 219.41000000]),
    "cam":         np.array([136.65940000, 143.03020000, 100.49700000]),
    "can":         np.array([100.79160000, 181.79580000, 193.73400000]),
    "cat":         np.array([67.01070000, 127.63300000, 117.45660000]),
    "driller":     np.array([229.47600000, 75.47140000, 208.00200000]),
    "duck":        np.array([104.42920000, 77.40760000, 85.69700000]),
    "eggbox":      np.array([150.18460000, 107.07500000,  69.24140000]),
    "glue":        np.array([36.72110000, 77.86600000, 172.81580000]),
    "holepuncher": np.array([100.88780000, 108.49700000, 90.80000000]),
    "iron":        np.array([258.22600000, 118.48210000, 141.13240000]),
    "lamp":        np.array([203.14600000, 117.75250000, 213.11600000]),
    "phone":       np.array([93.91810000, 147.43340000, 184.74740000]),
    }


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
                 LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES, preprocess=None,
                 postprocess=None, variances=[1.0, 1.0, 1.0, 1.0], draw=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.class_to_sizes = LINEMOD_OBJECT_SIZES
        self.camera_matrix = LINEMOD_CAMERA_MATRIX
        self.draw = draw
        if preprocess is None:
            self.preprocess = EfficientPosePreprocess(model)
        if postprocess is None:
            self.postprocess = EfficientPosePostprocess(
                model, class_names, score_thresh, nms_thresh)

        super(DetectAndEstimateSingleShot, self).__init__()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def _build_draw_pose6D(self, name_to_size, camera_parameter):
        name_to_draw = {}
        for name, object_sizes in name_to_size.items():
            draw = pr.DrawPose6D(object_sizes, camera_parameter)
            name_to_draw[name] = draw
        return name_to_draw

    def call(self, image):
        preprocessed_data = self.preprocess(image)
        preprocessed_image, image_scale, camera_parameter = preprocessed_data
        outputs = self.model(preprocessed_image)
        detections, (rotations, translations) = outputs
        detections = change_box_coordinates(detections)
        outputs = detections, (rotations, translations)
        boxes2D, poses6D = self.postprocess(
            outputs, image_scale, camera_parameter)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            self.draw_pose6D = self._build_draw_pose6D(
                self.class_to_sizes, self.camera_matrix)
            for box2D, pose6D in zip(boxes2D, poses6D):
                image = self.draw_pose6D[box2D.class_name](image, pose6D)
        return self.wrap(image, boxes2D, poses6D)


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
                 variances=[1.0, 1.0, 1.0, 1.0], class_arg=None):
        super(EfficientPosePostprocess, self).__init__()
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
        self.regress_translation = RegressTranslation(model.translation_priors)
        self.compute_tx_ty = ComputeTxTy()
        self.compute_selections = ComputeSelectedIndices()
        self.transform_rotations = pr.Scale(np.pi)
        self.to_pose_6D = ToPose6D(class_names)

    def call(self, output, image_scale, camera_parameter):
        detections, (rotations, translations) = output
        box_data = self.postprocess(detections)
        box_data = self.scale(box_data, 1 / image_scale)
        box_data_all = box_data
        box_data, class_labels = self.nms_per_class(box_data)
        box_data = self.merge_box_and_class(box_data, class_labels)
        box_data = self.filter_boxes(box_data)
        boxes2D = self.to_boxes2D(box_data)
        boxes2D = self.round_boxes(boxes2D)

        selected_indices = self.compute_selections(box_data_all, box_data)
        rotations = np.array(rotations[0, :, :])[selected_indices]
        rotations = self.transform_rotations(rotations)

        translation_xy_Tz = self.regress_translation(translations)
        translation = self.compute_tx_ty(translation_xy_Tz, camera_parameter)
        translations = translation[selected_indices]

        poses6D = self.to_pose_6D(box_data, rotations, translations)
        return boxes2D, poses6D


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
            model, names, score_thresh, nms_thresh,
            LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES, draw=draw)
