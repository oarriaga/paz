import numpy as np

from ..abstract import Processor, Pose6D
from ..backend.keypoints import solve_PNP
from ..backend.keypoints import LEVENBERG_MARQUARDT


class SolvePNP(Processor):
    """Calculates 6D pose from 3D points and 2D keypoints correspondences.

    # Arguments
        model_points: Numpy array of shape ``[num_points, 3]``.
            Model 3D points known in advance.
        camera: Instance of ''paz.backend.Camera'' containing as properties
            the ``camera_intrinsics`` a Numpy array of shape ``[3, 3]``
            usually calculated from the openCV ``calibrateCamera`` function,
            and the ``distortion`` a Numpy array of shape ``[5]`` in which the
            elements are usually obtained from the openCV
            ``calibrateCamera`` function.
        solver: Flag specifying solvers. Current solvers are:
            ``paz.processors.LEVENBERG_MARQUARDT`` and ``paz.processors.UPNP``.

    # Returns
        Instance from ``Pose6D`` message.
    """
    def __init__(self, points3D, camera, solver=LEVENBERG_MARQUARDT):
        super(SolvePNP, self).__init__()
        self.points3D = points3D
        self.camera = camera
        self.solver = solver
        self.num_keypoints = len(points3D)

    def call(self, keypoints):
        keypoints = keypoints[:, :2]
        keypoints = keypoints.astype(np.float64)
        keypoints = keypoints.reshape((self.num_keypoints, 1, 2))

        (success, rotation, translation) = solve_PNP(
            self.points3D, keypoints, self.camera, self.solver)

        return Pose6D.from_rotation_vector(rotation, translation)
