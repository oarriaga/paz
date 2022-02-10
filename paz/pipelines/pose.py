from ..abstract import Processor, SequentialProcessor
from .. import processors as pr

from .detection import HaarCascadeFrontalFace
from .keypoints import FaceKeypointNet2D32
import numpy as np


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
                three integers indicating the width, height and depth of the
                object e.g. {'PowerDrill': [30, 20, 10]}.
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
        self.draw_box = pr.DrawBoxes3D(camera, class_to_dimensions,
                                       thickness=thickness)
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
    def __init__(self, camera, offsets=[0, 0], radius=5, thickness=2):
        detect = HaarCascadeFrontalFace(draw=False)
        estimate_keypoints = FaceKeypointNet2D32(draw=False)
        """
                               4--------1
                              /|       /|
                             / |      / |
                            3--------2  |
                            |  8_____|__5
                            | /      | /
                            |/       |/
                            7--------6

                   Z (depth)
                  /
                 /_____X (width)
                 |
                 |
                 Y (height)
        """
        KEYPOINTS3D = np.array([
            [-220, 1138, 678],  # left--center-eye
            [+220, 1138, 678],  # right-center-eye
            [-131, 1107, 676],  # left--eye close to nose
            [-294, 1123, 610],  # left--eye close to ear
            [+131, 1107, 676],  # right-eye close to nose
            [+294, 1123, 610],  # right-eye close to ear
            [-106, 1224, 758],  # left--eyebrow close to nose
            [-375, 1208, 585],  # left--eyebrow close to ear
            [+106, 1224, 758],  # right-eyebrow close to nose
            [+375, 1208, 585],  # right-eyebrow close to ear
            [0.0, 919, 909],  # nose
            [-183, 683, 691],  # lefty-lip
            [+183, 683, 691],  # right-lip
            [0.0, 754, 826],  # up---lip
            [0.0, 645, 815],  # down-lip
        ])
        KEYPOINTS3D = KEYPOINTS3D - np.mean(KEYPOINTS3D, axis=0)
        super(HeadPoseKeypointNet2D32, self).__init__(
            detect, estimate_keypoints, camera, offsets,
            KEYPOINTS3D, {None: [900, 1200, 800]}, radius, thickness)
