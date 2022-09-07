import numpy as np

from ..abstract import Processor, Pose6D
from ..backend.keypoints import solve_PNP
from ..backend.keypoints import LEVENBERG_MARQUARDT
from ..backend.keypoints import solve_PnP_RANSAC


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


class SolveChangingObjectPnPRANSAC(Processor):
    """Returns rotation (Roc) and translation (Toc) vectors that transform
        3D points in object frame to camera frame.

                               O------------O
                              /|           /|
                             / |          / |
                            O------------O  |
                            |  |    z    |  |
                            |  O____|____|__O
                            |  /    |___y|  /   object
                            | /    /     | /  coordinates
                            |/    x      |/
                            O------------O
                                   ___
                   Z                |
                  /                 | Rco, Tco
                 /_____X     <------|
                 |
                 |    camera
                 Y  coordinates

    # Arguments
        object_points3D: Array (num_points, 3). Points 3D in object reference
            frame. Represented as (0) in image above.
        image_points2D: Array (num_points, 2). Points in 2D in camera UV space.
        camera_intrinsics: Array of shape (3, 3). Diagonal elements represent
            focal lenghts and last column the image center translation.
        inlier_threshold: Number of inliers for RANSAC method.
        num_iterations: Maximum number of iterations.

    # Returns
        Boolean indicating success, rotation vector in axis-angle form (3)
            and translation vector (3).
    """

    def __init__(self, camera_intrinsics, inlier_thresh=5, num_iterations=100):
        super(SolveChangingObjectPnPRANSAC, self).__init__()
        self.camera_intrinsics = camera_intrinsics
        self.inlier_thresh = inlier_thresh
        self.num_iterations = num_iterations
        self.MIN_REQUIRED_POINTS = 4

    def call(self, object_points3D, image_points2D):
        success, rotation_vector, translation = solve_PnP_RANSAC(
            object_points3D, image_points2D, self.camera_intrinsics,
            self.inlier_thresh, self.num_iterations)
        rotation_vector = np.squeeze(rotation_vector)
        return success, rotation_vector, translation


class Translation3DFromBoxWidth(Processor):
    """Computes 3D translation from box width and real width ratio.

    # Arguments
        camera: Instance of ''paz.backend.Camera'' containing as properties
            the ``camera_intrinsics`` a Numpy array of shape ``[3, 3]``
            usually calculated from the openCV ``calibrateCamera`` function,
            and the ``distortion`` a Numpy array of shape ``[5]`` in which the
            elements are usually obtained from the openCV
            ``calibrateCamera`` function.
        real_width: Real width of the predicted box2D.

    # Returns
        Array (num_boxes, 3) containing all 3D translations.
    """
    def __init__(self, camera, real_width=0.3):
        super(Translation3DFromBoxWidth, self).__init__()
        self.camera = camera
        self.real_width = real_width
        self.focal_length = self.camera.intrinsics[0, 0]
        self.u_camera_center = self.camera.intrinsics[0, 2]
        self.v_camera_center = self.camera.intrinsics[1, 2]

    def call(self, boxes2D):
        hands_center = []
        for box in boxes2D:
            u_box_center, v_box_center = box.center
            z_center = (self.real_width * self.focal_length) / box.width
            u = u_box_center - self.u_camera_center
            v = v_box_center - self.v_camera_center
            x_center = (z_center * u) / self.focal_length
            y_center = (z_center * v) / self.focal_length
            hands_center.append([x_center, y_center, z_center])
        return np.array(hands_center)
