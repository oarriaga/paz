from ..abstract import Processor, SequentialProcessor, Pose6D
from .. import processors as pr
from ..backend.keypoints import (points3D_to_RGB, build_cube_points3D,
                                 denormalize_keypoints2D)
from ..backend.groups.quaternion import rotation_vector_to_quaternion
from ..backend.image.draw import draw_points2D, draw_circle
from ..models import UNET_VGG16

from .masks import Pix2Points
from .detection import HaarCascadeFrontalFace
from .keypoints import FaceKeypointNet2D32
import numpy as np
from .detection import SSD300FAT
from tensorflow.keras.utils import get_file


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


class RGBMaskToPose6D(pr.Processor):
    """Predicts pose6D from an RGB mask

    # Arguments
        model: Keras segmentation model.
        object_sizes: Array (3) determining the (width, height, depth)
        camera: PAZ Camera with intrinsic matrix.
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized to original shape.
        class_name: Str indicating object name.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred points2D, points3D, pose6D and image.
    """
    def __init__(self, model, object_sizes, camera, epsilon=0.15,
                 resize=False, class_name=None, draw=True):
        super(RGBMaskToPose6D, self).__init__()
        self.model = model
        self.resize = resize
        self.object_sizes = object_sizes
        self.camera = camera
        self.epsilon = epsilon
        self.class_name = str(class_name) if class_name is None else class_name
        self.draw = draw

        self.predict_points = Pix2Points(
            self.model, self.object_sizes, self.epsilon, self.resize)
        self.predict_pose = pr.SolveChangingObjectPnPRANSAC(camera.intrinsics)
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw_pose6D = pr.DrawPose6D(self.cube_points3D,
                                         self.camera.intrinsics)

    def call(self, image, box2D=None):
        results = self.predict_points(image)
        points2D, points3D = results['points2D'], results['points3D']
        H, W = image.shape[:2]
        points2D = denormalize_keypoints2D(points2D, H, W)

        if box2D is not None:
            points2D = self.change_coordinates(points2D, box2D)
            self.class_name = box2D.class_name

        if len(points3D) > self.predict_pose.MIN_REQUIRED_POINTS:
            success, R, translation = self.predict_pose(points3D, points2D)
            if success:
                quaternion = rotation_vector_to_quaternion(R)
                pose6D = Pose6D(quaternion, translation, self.class_name)
            else:
                pose6D = None
        else:
            pose6D = None

        # box2D check required since change_coordinates goes outside (crop) img
        if (self.draw and (box2D is None) and (pose6D is not None)):
            colors = points3D_to_RGB(points3D, self.object_sizes)
            image = draw_points2D(image, points2D, colors)
            image = self.draw_pose6D(image, pose6D)
            results['image'] = image
        results['points2D'], results['pose6D'] = points2D, pose6D
        return results


class PIX2POSE(Processor):
    """Predicts pose6D from an RGB mask

    # Arguments
        detect: Function for estimating bounding boxes2D.
        estimate_pose: Function for estimating pose6D.
        offsets: Float between [0, 1] indicating ratio of increase of box2D.
        valid_class_names: List of strings indicating class names to be kept.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """

    def __init__(self, detect, estimate_pose, offsets, draw=True,
                 valid_class_names=None):
        super(PIX2POSE, self).__init__()
        self.detect = detect
        self.estimate_pose = estimate_pose

        self.postprocess_boxes = SequentialProcessor()
        self.postprocess_boxes.add(pr.UnpackDictionary(['boxes2D']))
        if valid_class_names is not None:
            self.postprocess_boxes.add(
                pr.FilterClassBoxes2D(valid_class_names))
        self.postprocess_boxes.add(pr.SquareBoxes2D())
        self.postprocess_boxes.add(pr.OffsetBoxes2D(offsets))

        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.unwrap = pr.UnwrapDictionary(['pose6D', 'points2D', 'points3D'])
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.object_sizes = self.estimate_pose.object_sizes
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw_pose6D = pr.DrawPose6D(
            self.cube_points3D, self.estimate_pose.camera.intrinsics)
        self.draw = draw

    def call(self, image):
        boxes2D = self.postprocess_boxes(self.detect(image))
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points2D, points3D = [], [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            results = self.estimate_pose(crop, box2D)
            pose6D, set_points2D, set_points3D = self.unwrap(results)
            points2D.append(set_points2D), points3D.append(set_points3D)
            poses6D.append(pose6D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            for set_points2D, set_points3D in zip(points2D, points3D):
                colors = points3D_to_RGB(set_points3D, self.object_sizes)
                # image = draw_points2D(image, set_points2D, colors)
                # NOTE: DECIDED to use for loop for better visualization
                for point2D, color in zip(set_points2D, colors):
                    R, G, B = color
                    draw_circle(image, point2D.astype(np.int64),
                                (int(R), int(G), int(B)))
            for pose6D in poses6D:
                image = self.draw_pose6D(image, pose6D)
        return self.wrap(image, boxes2D, poses6D)


class RGBMaskToPowerDrillPose6D(RGBMaskToPose6D):
    def __init__(self, camera, epsilon=0.15, resize=False, draw=True):
        model = UNET_VGG16(3, (128, 128, 3))
        URL = ('https://github.com/oarriaga/altamira-data/'
               'releases/download/v0.13/')
        name = 'UNET-VGG16_POWERDRILL_weights.hdf5'
        weights_path = get_file(name, URL + name, cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path)
        object_sizes = np.array([1840, 1870, 520])
        class_name = '035_power_drill'
        super(RGBMaskToPowerDrillPose6D, self).__init__(
            model, object_sizes, camera, epsilon, resize, class_name, draw)


class PIX2POSEPowerDrill(PIX2POSE):
    """PIX2POSE inference pipeline with SSD300 trained on FAT and UNET-VGG16
        trained with domain randomization for the object power drill.

    # Arguments
        score_thresh: Float between [0, 1] for object detector.
        nms_thresh: Float between [0, 1] indicating the non-maximum supression.
        epsilon: Float. Values below this value would be replaced by 0.
        offsets: List of 2 between [0, 1] indicating percentage increase of box
            dimensions.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    """
    def __init__(self, camera, score_thresh=0.50, nms_thresh=0.45,
                 epsilon=0.15, offsets=[0.5, 0.5], draw=True):
        detect = SSD300FAT(score_thresh, nms_thresh, draw=False)
        estimate_pose = RGBMaskToPowerDrillPose6D(camera, epsilon, draw=False)
        super(PIX2POSEPowerDrill, self).__init__(
            detect, estimate_pose, offsets, draw, ['035_power_drill'])
