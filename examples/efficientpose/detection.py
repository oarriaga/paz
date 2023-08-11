from efficientpose import EFFICIENTPOSEA
from paz.pipelines.detection import DetectSingleShotEfficientDet


def get_class_names(dataset_name='LINEMOD'):
    if dataset_name in ['LINEMOD']:
        class_names = ['ape', 'can', 'cat', 'driller', 'duck',
                       'eggbox', 'glue', 'holepuncher']

    return class_names


class EFFICIENTPOSEALINEMOD(DetectSingleShotEfficientDet):
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