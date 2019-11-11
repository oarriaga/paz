import numpy as np

from ..core import Processor
from ..core import Pose6D
from ..core import ops


class SolvePNP(Processor):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.
    # Arguments
        model_points: Numpy array of shape (num_points, 3).
            Model 3D points known in advance.
        camera intrinsics: Numpy array of shape (3, 3) calculated from
        the openCV calibrateCamera function
        distortion: Numpy array of shape of 5 elements calculated from
        the openCV calibrateCamera function
    # Returns
        Creates a new topic ``pose6D`` with a Pose6D message.
    """
    def __init__(self, model_points, camera_intrinsics, distortion):
        super(SolvePNP, self).__init__()
        self.model_points = model_points
        self.camera_intrinsics = camera_intrinsics
        self.distortion = distortion
        self.num_keypoints = len(model_points)

    def call(self, kwargs):
        keypoints = kwargs['keypoints'][:, :2]
        keypoints = keypoints.astype(np.float64)
        keypoints = keypoints.reshape((self.num_keypoints, 1, 2))

        (success, rotation, translation) = ops.solve_PNP(
            self.model_points, keypoints, self.camera_intrinsics,
            ops.UPNP, self.distortion)

        if 'box2D' in kwargs:
            class_name = kwargs['box2D'].class_name
        else:
            class_name = None
        pose6D = Pose6D.from_rotation_vector(rotation, translation, class_name)
        kwargs['pose6D'] = pose6D
        return kwargs
