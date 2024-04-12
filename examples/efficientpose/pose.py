from paz.pipelines.pose import EstimateEfficientPose
from linemod import (LINEMOD_CAMERA_MATRIX, LINEMOD_OBJECT_SIZES,
                     RGB_LINEMOD_MEAN)
from paz.models.pose_estimation.efficientpose import EfficientPosePhi0
from processors import ComputeTxTyTz, RegressTranslation
from anchors import build_translation_anchors


class EfficientPosePhi0LinemodDriller(EstimateEfficientPose):
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
        model = EfficientPosePhi0(build_translation_anchors,
                                  num_classes=len(names), base_weights='COCO',
                                  head_weights=None)
        regress_translation = RegressTranslation(model.translation_priors)
        compute_tx_ty_tz = ComputeTxTyTz()
        super(EfficientPosePhi0LinemodDriller, self).__init__(
            model, names, score_thresh, nms_thresh,
            LINEMOD_OBJECT_SIZES, RGB_LINEMOD_MEAN, LINEMOD_CAMERA_MATRIX,
            regress_translation, compute_tx_ty_tz, show_boxes2D=show_boxes2D,
            show_poses6D=show_poses6D)
