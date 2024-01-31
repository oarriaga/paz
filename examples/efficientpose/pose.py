from paz.pipelines.pose import DetectAndEstimateEfficientPose
from paz.datasets.linemod import LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES
from paz.models.pose_estimation.efficientpose import EfficientPosePhi0


class EfficientPosePhi0LinemodDriller(DetectAndEstimateEfficientPose):
    """Inference pipeline with EfficientPose phi=0 trained on Linemod.

    # Arguments
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        show_boxes2D: Boolean. If ``True`` prediction
            are drawn in the returned image.
        show_poses6D: Boolean. If ``True`` estimated poses
            are drawn in the returned image.

     # References
        [EfficientPose: An efficient, accurate and scalable end-to-end
        6D multi object pose estimation approach](
            https://arxiv.org/pdf/2011.04307.pdf)
    """
    def __init__(self, score_thresh=0.60, nms_thresh=0.45,
                 show_boxes2D=False, show_poses6D=True):
        names = ['background', 'driller']
        model = EfficientPosePhi0(num_classes=len(names), base_weights='COCO',
                                  head_weights=None)
        super(EfficientPosePhi0LinemodDriller, self).__init__(
            model, names, score_thresh, nms_thresh,
            LINEMOD_OBJECT_SIZES, LINEMOD_CAMERA_MATRIX,
            show_boxes2D=show_boxes2D, show_poses6D=show_poses6D)
