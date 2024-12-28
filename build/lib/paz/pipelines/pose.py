from ..abstract import Processor, SequentialProcessor
from .. import processors as pr

from .detection import HaarCascadeFrontalFace
from .keypoints import FaceKeypointNet2D32
import numpy as np


FACE_KEYPOINTNET3D = np.array([
    [-220, 678, 1138],  # left--center-eye
    [+220, 678, 1138],  # right-center-eye
    [-131, 676, 1107],  # left--eye close to nose
    [-294, 610, 1123],  # left--eye close to ear
    [+131, 676, 1107],  # right-eye close to nose
    [+294, 610, 1123],  # right-eye close to ear
    [-106, 758, 1224],  # left--eyebrow close to nose
    [-375, 585, 1208],  # left--eyebrow close to ear
    [+106, 758, 1224],  # right-eyebrow close to nose
    [+375, 585, 1208],  # right-eyebrow close to ear
    [0.0, 909, 919],  # nose
    [-183, 691, 683],  # lefty-lip
    [+183, 691, 683],  # right-lip
    [0.0, 826, 754],  # up---lip
    [0.0, 815, 645],  # down-lip
])

FACE_KEYPOINTNET3D = FACE_KEYPOINTNET3D - np.mean(FACE_KEYPOINTNET3D, axis=0)


class EstimatePoseKeypoints(Processor):
    def __init__(self, detect, estimate_keypoints, camera, offsets,
                 model_points, class_to_dimensions, radius=3, thickness=1):
        """Pose estimation pipeline using keypoints.

        # Arguments
            detect: Function that outputs a dictionary with a key
                ``boxes2D`` having a list of ``Box2D`` messages.
            estimate_keypoints: Function that outputs a dictionary
                with a key ``keypoints`` with numpy array as value
            camera: Instance of ``paz.backend.camera.Camera`` with
                camera intrinsics.
            offsets: List of floats indicating the scaled offset to
                be added to the ``Box2D`` coordinates.
            model_points: Numpy array of shape ``(num_keypoints, 3)``
                indicating the 3D coordinates of the predicted keypoints
                from the ``esimate_keypoints`` function.
            class_to_dimensions: Dictionary with keys being the class labels
                of the predicted ``Box2D`` messages and the values a list of
                two integers indicating the height and width of the object.
                e.g. {'PowerDrill': [30, 20]}.
            radius: Int. radius of keypoint to be drawn.
            thickness: Int. thickness of 3D box.

        # Returns
            A function that takes an RGB image and outputs the following
            inferences as keys of a dictionary:
                ``image``, ``boxes2D``, ``keypoints`` and ``poses6D``.
        """
        super(EstimatePoseKeypoints, self).__init__()
        self.num_keypoints = estimate_keypoints.num_keypoints
        self.detect = detect
        self.estimate_keypoints = estimate_keypoints
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.solve_PNP = pr.SolvePNP(model_points, camera)
        self.draw_keypoints = pr.DrawKeypoints2D(self.num_keypoints, radius)
        self.draw_box = pr.DrawBoxes3D(camera, class_to_dimensions, thickness)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'keypoints', 'poses6D'])

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, keypoints2D = [], []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)['keypoints']
            keypoints = self.change_coordinates(keypoints, box2D)
            pose6D = self.solve_PNP(keypoints)
            image = self.draw_keypoints(image, keypoints)
            image = self.draw_box(image, pose6D)
            keypoints2D.append(keypoints)
            poses6D.append(pose6D)
        return self.wrap(image, boxes2D, keypoints2D, poses6D)


class HeadPoseKeypointNet2D32(EstimatePoseKeypoints):
    """Head pose estimation pipeline using a ``HaarCascade`` face detector
        and a pre-trained ``KeypointNet2D`` estimation model.

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera`` with
                camera intrinsics.
            offsets: List of floats indicating the scaled offset to
                be added to the ``Box2D`` coordinates.
            radius: Int. radius of keypoint to be drawn.

        # Example
            ``` python
            from paz.pipelines import HeadPoseKeypointNet2D32

            estimate_pose = HeadPoseKeypointNet2D32()

            # apply directly to an image (numpy-array)
            inferences = estimate_pose(image)
            ```

        # Returns
            A function that takes an RGB image and outputs the following
            inferences as keys of a dictionary:
                ``image``, ``boxes2D``, ``keypoints`` and ``poses6D``.
        """
    def __init__(self, camera, offsets=[0, 0], radius=3, thickness=1):
        detect = HaarCascadeFrontalFace(draw=False)
        estimate_keypoints = FaceKeypointNet2D32(draw=False)
        super(HeadPoseKeypointNet2D32, self).__init__(
            detect, estimate_keypoints, camera, offsets,
            FACE_KEYPOINTNET3D, {None: [900.0, 600.0]}, radius, thickness)
