from paz.abstract import Processor
import paz.processors as pr
from data_utils import unnormalize_data, load_joints_2D
from data import mean2D, stdd2D, data_mean3D, \
    data_std3D, dim_to_use3D
from backend import standardize


class SIMPLEBASELINES(Processor):
    def __init__(self, estimate_keypoints_2D, estimate_keypoints_3D):
        self.estimate_keypoints_2D = estimate_keypoints_2D
        self.estimate_keypoints_3D = estimate_keypoints_3D
        self.wrap = pr.WrapOutput(['keypoints2D', 'keypoints3D'])

    def call(self, image):
        inferences2D = self.estimate_keypoints_2D(image)
        keypoints2D = inferences2D['keypoints']
        keypoints2D = load_joints_2D(keypoints2D)
        norm_data = standardize(keypoints2D, mean2D, stdd2D)
        keypoints3D = self.estimate_keypoints_3D.predict(norm_data)
        keypoints3D = unnormalize_data(keypoints3D, data_mean3D, data_std3D,
                                       dim_to_use3D)
        return self.wrap(keypoints2D, keypoints3D)
