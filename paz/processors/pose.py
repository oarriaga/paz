import numpy as np

from ..abstract import Processor, Pose6D
from ..backend.keypoints import solve_PNP
from ..backend.keypoints import LEVENBERG_MARQUARDT
from ..backend.keypoints import solve_PnP_RANSAC
from ..backend.poses import match_poses
from ..backend.poses import rotation_matrix_to_axis_angle
from ..backend.poses import concatenate_poses
from ..backend.poses import concatenate_scale
from ..backend.poses import augment_pose_6D


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


class MatchPoses(Processor):
    """Match prior boxes with ground truth poses.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        iou: Float in [0, 1]. Intersection over union in which prior
            boxes will be considered positive. A positive box is box
            with a class different than `background`.
    """
    def __init__(self, prior_boxes, iou=.5):
        self.prior_boxes = prior_boxes
        self.iou = iou
        super(MatchPoses, self).__init__()

    def call(self, boxes, poses):
        return match_poses(boxes, poses, self.prior_boxes, self.iou)


class RotationMatrixToAxisAngle(Processor):
    """Computes axis angle rotation vector from a rotation matrix.

    # Arguments:
        num_pose_dims: Int, number of dimensions of pose.

    # Returns:
        transformed_rotations: Array of shape (5,)
            containing transformed rotation.
    """
    def __init__(self, num_pose_dims):
        self.num_pose_dims = num_pose_dims
        super(RotationMatrixToAxisAngle, self).__init__()

    def call(self, rotations):
        return rotation_matrix_to_axis_angle(rotations, self.num_pose_dims)


class ConcatenatePoses(Processor):
    """Concatenates rotations and translations into a single array.

    # Returns:
        poses_combined: Array of shape `(num_boxes, 10)`
            containing the transformed rotation.
    """
    def __init__(self):
        super(ConcatenatePoses, self).__init__()

    def call(self, rotations, translations):
        return concatenate_poses(rotations, translations)


class ConcatenateScale(Processor):
    """Concatenates poses with image scale into a single array.

    # Returns:
        poses_combined: Array of shape `(num_prior_boxes, 11)`
            containing the transformed rotation.
    """
    def __init__(self):
        super(ConcatenateScale, self).__init__()

    def call(self, poses, scale):
        return concatenate_scale(poses, scale)


class AugmentPose6D(Processor):
    """Augment images, boxes, rotation and translation vector
    for pose estimation.

    # Arguments
        camera_matrix: Array with camera matrix of shape `(3, 3)`.
        scale_min: Float, minimum value to scale image.
        scale_max: Float, maximum value to scale image.
        angle_min: Int, minimum degree to rotate image.
        angle_max: Int, maximum degree to rotate image.
        probability: Float, probability of data transformation.
        mask_value: Int, pixel gray value of foreground in mask image.
        input_size: Int, input image size of the model.
    """
    def __init__(self, camera_matrix, scale_min=0.7, scale_max=1.3,
                 angle_min=0, angle_max=360, probability=0.5,
                 mask_value=255, input_size=512):
        self.camera_matrix = camera_matrix
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.probability = probability
        self.mask_value = mask_value
        self.input_size = input_size
        super(AugmentPose6D, self).__init__()

    def call(self, image, boxes, rotation, translation_raw, mask):
        if np.random.rand() < self.probability:
            augmented_data = augment_pose_6D(
                image, boxes, rotation, translation_raw, mask, self.scale_min,
                self.scale_max, self.angle_min, self.angle_max,
                self.mask_value, self.input_size, self.camera_matrix)
        else:
            augmented_data = image, boxes, rotation, translation_raw, mask
        return augmented_data


class ToPose6D(Processor):
    """Transforms poses i.e rotations and
    translations into `Pose6D` messages.

    # Arguments
        class_names: List of class names ordered with respect
            to the class indices from the dataset ``boxes``.
        one_hot_encoded: Bool, indicating if scores are one hot vectors.
        default_score: Float, score to set.
        default_class: Str, class to set.
        box_method: Int, method to convert boxes to ``Boxes2D``.

    # Properties
        one_hot_encoded: Bool.
        box_processor: Callable.

    # Methods
        call()
    """
    def __init__(
            self, class_names=None, one_hot_encoded=False,
            default_score=1.0, default_class=None, box_method=0):
        if class_names is not None:
            arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        method_to_processor = {
            0: BoxesWithOneHotVectorsToPose6D(arg_to_class),
            1: BoxesToPose6D(default_score, default_class),
            2: BoxesWithClassArgToPose6D(arg_to_class, default_score)}
        self.pose_processor = method_to_processor[box_method]
        super(ToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        return self.pose_processor(box_data, rotations, translations)


class BoxesWithOneHotVectorsToPose6D(Processor):
    """Transforms poses into `Pose6D` messages
    given boxes with scores as one hot vectors.

    # Arguments
        arg_to_class: List, of classes.

    # Properties
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class):
        self.arg_to_class = arg_to_class
        super(BoxesWithOneHotVectorsToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        poses6D = []
        for box, rotation, translation in zip(box_data, rotations,
                                              translations):
            class_scores = box[4:]
            class_arg = np.argmax(class_scores)
            class_name = self.arg_to_class[class_arg]
            poses6D.append(Pose6D.from_rotation_vector(rotation, translation,
                                                       class_name))
        return poses6D


class BoxesToPose6D(Processor):
    """Transforms poses into `Pose6D` messages
    given no class names and score.

    # Arguments
        default_score: Float, score to set.
        default_class: Str, class to set.

    # Properties
        default_score: Float.
        default_class: Str.

    # Methods
        call()
    """
    def __init__(self, default_score=1.0, default_class=None):
        self.default_score = default_score
        self.default_class = default_class
        super(BoxesToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        poses6D = []
        for box, rotation, translation in zip(box_data, rotations,
                                              translations):
            poses6D.append(Pose6D.from_rotation_vector(rotation, translation,
                                                       self.default_class))
        return poses6D


class BoxesWithClassArgToPose6D(Processor):
    """Transforms poses into `Pose6D` messages
    given boxes with class argument.

    # Arguments
        default_score: Float, score to set.
        arg_to_class: List, of classes.

    # Properties
        default_score: Float.
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class, default_score=1.0):
        self.default_score = default_score
        self.arg_to_class = arg_to_class
        super(BoxesWithClassArgToPose6D, self).__init__()

    def call(self, box_data, rotations, translations):
        poses6D = []
        for box, rotation, translation in zip(box_data, rotations,
                                              translations):
            class_name = self.arg_to_class[box[-1]]
            poses6D.append(Pose6D.from_rotation_vector(rotation, translation,
                                                       class_name))
        return poses6D
