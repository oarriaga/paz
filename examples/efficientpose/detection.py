from efficientpose import EFFICIENTPOSEA
from paz.abstract import Processor, SequentialProcessor
from paz.pipelines import SSDPreprocess
import paz.processors as pr


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
            self.preprocess = SSDPreprocess(model)
        if postprocess is None:
            self.postprocess = EfficientPosePostprocess(
                model, class_names, score_thresh, nms_thresh)

        super(DetectAndEstimateSingleShot, self).__init__()
        self.denormalize = pr.DenormalizeBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        preprocessed_image = self.preprocess(image)
        outputs = self.model(preprocessed_image)
        detections, pose = outputs
        boxes2D = self.postprocess(detections)
        boxes2D = self.denormalize(image, boxes2D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


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


class EfficientPosePostprocess(SequentialProcessor):
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
        super(EfficientPosePostprocess, self).__init__()
        self.add(pr.Squeeze(axis=None))
        self.add(pr.DecodeBoxes(model.prior_boxes, variances))
        self.add(pr.NonMaximumSuppressionPerClass(nms_thresh))
        self.add(pr.MergeNMSBoxWithClass())
        self.add(pr.FilterBoxes(class_names, score_thresh))
        self.add(pr.ToBoxes2D(class_names, box_method))
