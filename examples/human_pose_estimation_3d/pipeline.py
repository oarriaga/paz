from paz.abstract import Processor
import paz.processors as pr
from paz.applications import HigherHRNetHumanPose2D
import numpy as np
from keypoints_processors import SimpleBaselines3D
from human36m import data_mean2D, data_stdev2D, data_mean3D, data_stdev3D, \
    dim_to_use3D


class SIMPLEBASELINES(Processor):
    def __init__(self, estimate_keypoints_3D, args_to_mean,
                 h36m_to_coco_joints2D):
        """
        # Arguments
            estimate_keypoints_3D: 3D simple baseline model
            args_to_mean: keypoints indices
            h36m_to_coco_joints2D: h36m joints indices

        # Returns
            wrapped keypoints2D, keypoints3D
        """
        self.estimate_keypoints_2D = HigherHRNetHumanPose2D()
        self.estimate_keypoints_3D = estimate_keypoints_3D
        self.baseline_model = SimpleBaselines3D(self.estimate_keypoints_3D,
                                                data_mean2D, data_stdev2D,
                                                data_mean3D, data_stdev3D,
                                                dim_to_use3D, args_to_mean,
                                                h36m_to_coco_joints2D)
        self.wrap = pr.WrapOutput(['keypoints2D', 'keypoints3D'])

    def call(self, image):
        inferences2D = self.estimate_keypoints_2D(image)
        keypoints2D = inferences2D['keypoints']
        keypoints2D, keypoints3D = self.baseline_model(np.array(keypoints2D))
        return self.wrap(keypoints2D, keypoints3D)
