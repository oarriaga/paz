import numpy as np
from tensorflow.keras.utils import get_file

from .. import processors as pr
from ..abstract import Processor, SequentialProcessor, Pose6D
from ..models import UNET_VGG16
from ..backend.image.draw import draw_points2D, points3D_to_RGB
from ..backend.standard import append_lists
from ..backend.keypoints import (
    translate_points2D_origin, denormalize_keypoints2D)

from .masks import Pix2Points
from .detection import HaarCascadeFrontalFace
from .keypoints import FaceKeypointNet2D32
from .detection import SSD300FAT, PostprocessBoxes2D, PreprocessBoxes


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


class SingleInstancePIX2POSE6D(Processor):
    """Predicts a single pose6D from an image. Optionally if a box2D message is
        given it translates the predicted points2D to new origin located at
        box2D top-left corner.

    # Arguments
        model: Keras segmentation model.
        object_sizes: Array (3) determining the (width, height, depth)
        camera: PAZ Camera with intrinsic matrix.
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        class_name: Str indicating object name.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred points2D, points3D, pose6D and image.
    """
    def __init__(self, model, object_sizes, camera,
                 epsilon=0.15, resize=False, class_name=None, draw=True):
        super(SingleInstancePIX2POSE6D, self).__init__()
        self.camera = camera
        self.pix2points = Pix2Points(model, object_sizes, epsilon, resize)
        self.solvePnP = pr.SolveChangingObjectPnPRANSAC(self.camera.intrinsics)
        self.draw_pose6D = pr.DrawPose6D(object_sizes, self.camera.intrinsics)
        self.wrap = pr.WrapOutput(['image', 'points2D', 'points3D', 'pose6D'])
        self.class_name = str(class_name)
        self.object_sizes = object_sizes
        self.draw = draw

    def call(self, image, box2D=None):
        inferences = self.pix2points(image)
        points2D = inferences['points2D']
        points3D = inferences['points3D']
        points2D = denormalize_keypoints2D(points2D, *image.shape[:2])
        if box2D is not None:
            points2D = translate_points2D_origin(points2D, box2D.coordinates)
            self.class_name = box2D.class_name
        pose6D = None
        if len(points3D) > self.solvePnP.MIN_REQUIRED_POINTS:
            success, R, T = self.solvePnP(points3D, points2D)
            if success:
                pose6D = Pose6D.from_rotation_vector(R, T, self.class_name)
        if (self.draw and (box2D is None) and (pose6D is not None)):
            colors = points3D_to_RGB(points3D, self.object_sizes)
            image = draw_points2D(image, points2D, colors)
            image = self.draw_pose6D(image, pose6D)
        inferences = self.wrap(image, points2D, points3D, pose6D)
        return inferences


class MultiInstancePIX2POSE6D(Processor):
    """Predicts poses6D of multiple instances the same object from an image.

    # Arguments
        estimate_pose: Function that takes as input an image and outputs a
            dictionary with points2D, points3D and pose6D messages e.g
            SingleInstancePIX2POSE6D
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        camera: PAZ Camera with intrinsic matrix.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, estimate_pose, offsets, camera=None, draw=True):
        super(MultiInstancePIX2POSE6D, self).__init__()
        self.draw = draw
        self.estimate_pose = estimate_pose
        self.object_sizes = self.estimate_pose.object_sizes
        self.camera = self.estimate_pose.camera if camera is None else camera
        valid_names = [self.estimate_pose.class_name]
        self.postprocess_boxes = PostprocessBoxes2D(offsets, valid_names)

        self.append_values = pr.AppendValues(
            ['pose6D', 'points2D', 'points3D'])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.draw_RGBmask = pr.DrawRGBMasks(self.object_sizes)
        self.draw_boxes2D = pr.DrawBoxes2D(valid_names, colors=[[0, 255, 0]])
        self.draw_poses6D = pr.DrawPoses6D(
            self.object_sizes, camera.intrinsics)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

    def call(self, image, boxes2D):
        boxes2D = self.postprocess_boxes(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points2D, points3D = [], [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            inferences = self.estimate_pose(crop, box2D)
            self.append_values(inferences, [poses6D, points2D, points3D])
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            image = self.draw_RGBmask(image, points2D, points3D)
            image = self.draw_poses6D(image, poses6D)
        return self.wrap(image, boxes2D, poses6D)


class SinglePowerDrillPIX2POSE6D(SingleInstancePIX2POSE6D):
    """Predicts the pose6D of the YCB 035_power_drill object from an image.
        Optionally if a box2D message is given it translates the predicted
        points2D to new origin located at box2D top-left corner.

    # Arguments
        camera: PAZ Camera with intrinsic matrix.
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred points2D, points3D, pose6D and image.
    """
    def __init__(self, camera, epsilon=0.15, resize=False, draw=True):
        model = UNET_VGG16(3, (128, 128, 3))
        URL = ('https://github.com/oarriaga/altamira-data/'
               'releases/download/v0.13/')
        name = 'UNET-VGG16_POWERDRILL_weights.hdf5'
        weights_path = get_file(name, URL + name, cache_subdir='paz/models')
        print('Loading %s model weights' % weights_path)
        model.load_weights(weights_path)
        object_sizes = np.array([1840, 1870, 520]) / 10000
        class_name = '035_power_drill'
        super(SinglePowerDrillPIX2POSE6D, self).__init__(
            model, object_sizes, camera, epsilon, resize, class_name, draw)


class MultiPowerDrillPIX2POSE6D(MultiInstancePIX2POSE6D):
    """Predicts poses6D of multiple instances the YCB 035_power_drill object
        from an image.

    # Arguments
        camera: PAZ Camera with intrinsic matrix.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, camera, offsets, epsilon=0.15, resize=False, draw=True):
        estimate_pose = SinglePowerDrillPIX2POSE6D(
            camera, epsilon, resize, draw=False)
        super(MultiPowerDrillPIX2POSE6D, self).__init__(
            estimate_pose, offsets, camera, draw)


class PIX2POSEPowerDrill(Processor):
    """PIX2POSE inference pipeline with SSD300 trained on FAT and UNET-VGG16
        trained with domain randomization for the YCB object 035_power_drill.

    # Arguments
        score_thresh: Float between [0, 1] for object detector.
        nms_thresh: Float between [0, 1] indicating the non-maximum supression.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        epsilon: Float. Values below this value would be replaced by 0.
        draw: Boolean. If ``True`` prediction are drawn in the returned image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, camera, score_thresh=0.50, nms_thresh=0.45,
                 offsets=[0.5, 0.5], epsilon=0.15, resize=False, draw=True):
        self.detect = SSD300FAT(score_thresh, nms_thresh, draw=False)
        self.estimate_pose = MultiPowerDrillPIX2POSE6D(
            camera, offsets, epsilon, resize, draw)

    def call(self, image):
        return self.estimate_pose(image, self.detect(image)['boxes2D'])


class MultiInstanceMultiClassPIX2POSE6D(Processor):
    """Predicts poses6D of multiple instances of multiple objects from an image

    # Arguments
        detect: Function that takes as input an image and outputs a dictionary
            containing Boxes2D messages.
        name_to_model: Dictionary with class name as key and as value a
            Keras segmentation model.
        name_to_size: Dictionary with class name as key and as value the
            object sizes.
        camera: PAZ Camera with intrinsic matrix.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, detect, name_to_model, name_to_size, camera, offsets,
                 epsilon=0.15, resize=False, draw=True):
        super(MultiInstanceMultiClassPIX2POSE6D, self).__init__()
        if set(name_to_model.keys()) != set(name_to_size.keys()):
            raise ValueError('models and sizes must have same class names')
        self.detect = detect
        self.name_to_pix2points = self._build_pix2points(
            name_to_model, name_to_size, epsilon, resize)
        valid_names = list(self.name_to_model.keys())
        self.postprocess_boxes = PostprocessBoxes2D(offsets, valid_names)
        self.draw_boxes2D = pr.DrawBoxes2D(valid_names)
        self.draw_RGBmask = self._build_draw_RGBmask(name_to_size)
        self.draw_pose6D = self._build_draw_pose6D(name_to_size, camera)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'points3D', 'poses6D'])
        self.solvePnP = pr.SolveChangingObjectPnPRANSAC(camera.intrinsics)
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.draw = draw

    def _build_pix2points(self, name_to_model, name_to_size, epsilon, resize):
        name_to_pix2points = {}
        print(name_to_model)
        for name, model in name_to_model.items():
            pix2points = Pix2Points(model, name_to_size[name], epsilon, resize)
            name_to_pix2points[name] = pix2points
        return name_to_pix2points

    def _build_draw_pose6D(self, name_to_size, camera):
        name_to_draw = {}
        for name, object_sizes in name_to_size.items():
            draw = pr.DrawPose6D(object_sizes, camera.intrinsics)
            name_to_draw[name] = draw
        return name_to_draw

    def _build_draw_RGBmask(self, name_to_size):
        name_to_draw = {}
        for name, object_sizes in name_to_size.items():
            draw = pr.DrawRGBMask(object_sizes)
            name_to_draw[name] = draw
        return name_to_draw

    def estimate_pose(self, image, box2D):
        inferences = self.name_to_pix2points[box2D.class_name](image)
        points2D = inferences['points2D']
        points3D = inferences['points3D']
        points2D = denormalize_keypoints2D(points2D, *image.shape[:2])
        points2D = translate_points2D_origin(points2D, box2D.coordinates)
        pose6D = None
        if len(points3D) > self.solvePnP.MIN_REQUIRED_POINTS:
            success, R, T = self.solvePnP(points3D, points2D)
            if success:
                pose6D = Pose6D.from_rotation_vector(R, T, box2D.class_name)
        return points2D, points3D, pose6D

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.postprocess_boxes(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        points2D, points3D, poses6D = [], [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            inferences = self.estimate_pose(crop, box2D)
            append_lists(inferences, [points2D, points3D, poses6D])
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            for box2D, pose6D in zip(boxes2D, poses6D):
                name = box2D.class_name
                image = self.draw_pose6D[name](image, pose6D)
            for box2D, p2D, p3D in zip(boxes2D, points2D, points3D):
                image = self.draw_RGBmask[name](image, p2D, p3D)
        return self.wrap(image, boxes2D, points3D, poses6D)


class PIX2YCBTools6D(MultiInstanceMultiClassPIX2POSE6D):
    """Predicts poses6D of multiple instances of the YCB tools:
        '035_power_drill', '051_large_clamp', '037_scissors'

    # Arguments
        camera: PAZ Camera with intrinsic matrix.
        score_thresh: Float between [0, 1] for filtering Boxes2D.
        nsm_thresh: Float between [0, 1] non-maximum-supression filtering.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        epsilon: Float. Values below this value would be replaced by 0.
        resize: Boolean. If True RGB mask is resized before computing PnP.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, camera, score_thresh=0.45, nms_thresh=0.15,
                 offsets=[0.25, 0.25], epsilon=0.15, resize=False, draw=True):

        self.detect = SSD300FAT(score_thresh, nms_thresh, draw=False)
        self.name_to_sizes = self._build_name_to_sizes()
        self.name_to_model = self._build_name_to_model()
        super(PIX2YCBTools6D, self).__init__(
            self.detect, self.name_to_model, self.name_to_sizes, camera,
            offsets, epsilon, resize, draw)

    def _build_name_to_model(self):
        URL = ('https://github.com/oarriaga/altamira-data/'
               'releases/download/v0.13/')

        UNET_power_drill = UNET_VGG16(3, (128, 128, 3))
        name = 'UNET-VGG16_POWERDRILL_weights.hdf5'
        weights_path = get_file(name, URL + name, cache_subdir='paz/models')
        UNET_power_drill.load_weights(weights_path)

        UNET_large_clamp = UNET_VGG16(3, (128, 128, 3))
        name = 'UNET-VGG16_LARGE-CLAMP_weights.hdf5'
        weights_path = get_file(name, URL + name, cache_subdir='paz/models')
        UNET_large_clamp.load_weights(weights_path)

        UNET_scissors = UNET_VGG16(3, (128, 128, 3))
        name = 'UNET-VGG16_SCISSORS_weights.hdf5'
        weights_path = get_file(name, URL + name, cache_subdir='paz/models')
        UNET_scissors.load_weights(weights_path)

        name_to_model = {'035_power_drill': UNET_power_drill,
                         '051_large_clamp': UNET_large_clamp,
                         '037_scissors': UNET_scissors
                         }
        return name_to_model

    def _build_name_to_sizes(self):
        name_to_sizes = {
            '035_power_drill': np.array([1840, 1874, 572]) / 10000,
            '051_large_clamp': np.array([2022, 1652, 362]) / 10000,
            '037_scissors': np.array([960, 2014, 156]) / 10000
        }
        return name_to_sizes


class AugmentColor(SequentialProcessor):
    def __init__(self):
        super(AugmentColor, self).__init__()
        self.add(pr.EqualizeHistogram())
        self.add(pr.InvertColors())
        self.add(pr.Posterize())
        self.add(pr.Solarize())
        self.add(pr.RandomSaturation(1.0))
        self.add(pr.RandomContrast(1.0))
        self.add(pr.RandomBrightness(1.0))
        self.add(pr.SharpenImage())
        self.add(pr.Cutout())
        self.add(pr.RandomImageBlur())
        self.add(pr.RandomGaussianBlur())
        self.add(pr.AddGaussianNoise())


class AugmentEfficientPose(SequentialProcessor):
    """Augment images, boxes and poses for pose estimation.

    # Arguments
        model: Keras model.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would
            take place depending on the flag.
        num_classes: Int, specifying the number of classes to train.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        camera_matrix: Array with camera matrix of shape `(3, 3)`.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
        probability: Float indicating the probability
            of data augmentation.
        num_pose_dims: Int, number of dimensions for pose.
    """
    def __init__(self, model, mean, camera_matrix, split=pr.TRAIN,
                 num_classes=8, size=512, IOU=.5,
                 variances=[0.1, 0.1, 0.2, 0.2], probability=0.5,
                 num_pose_dims=3):
        super(AugmentEfficientPose, self).__init__()
        self.augment_color = AugmentColor()
        self.augment_pose_6D = pr.AugmentPose6D(
            camera_matrix, probability=probability, input_size=size)
        self.preprocess_image = EfficientPosePreprocess(model, mean)

        # box processors
        self.scale_boxes = pr.ScaleBox()
        args = (num_classes, model.prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        rotation_matrix_to_axis_angle = pr.RotationMatrixToAxisAngle(
            num_pose_dims)
        # pose processors
        self.match_poses = pr.MatchPoses(model.prior_boxes)
        self.concatenate_poses = pr.ConcatenatePoses()
        self.concatenate_scale = pr.ConcatenateScale()

        # pipeline
        self.add(pr.UnpackDictionary(['image', 'boxes', 'rotation',
                                      'translation_raw', 'class', 'mask']))
        self.add(pr.ControlMap(pr.LoadImage(), [0], [0]))
        self.add(pr.ControlMap(pr.LoadImage(), [5], [5]))
        if split == pr.TRAIN:
            self.add(pr.ControlMap(self.augment_color, [0], [0]))
            self.add(pr.ControlMap(self.augment_pose_6D, [0, 1, 2, 3, 5],
                                   [0, 1, 2, 3, 5]))
        self.add(pr.ControlMap(self.preprocess_image, [0], [0, 1, 2]))
        self.add(pr.ControlMap(self.scale_boxes, [2, 1], [2], keep={1: 1}))
        self.add(pr.ControlMap(self.preprocess_boxes, [3], [4], keep={3: 3}))
        self.add(pr.ControlMap(rotation_matrix_to_axis_angle, [2], [2]))
        self.add(pr.ControlMap(self.match_poses, [3, 2], [2], keep={3: 3}))
        self.add(pr.ControlMap(self.match_poses, [3, 4], [6], keep={3: 3}))
        self.add(pr.ControlMap(self.concatenate_poses, [2, 7], [7],
                               keep={2: 2}))
        self.add(pr.ControlMap(self.concatenate_scale, [7, 1], [7]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {3: {'boxes': [len(model.prior_boxes), 4 + num_classes]},
             6: {'transformation': [len(model.prior_boxes),
                                    3 * num_pose_dims + 2]}}))


class EfficientPosePreprocess(Processor):
    """Preprocessing pipeline for EfficientPose.

    # Arguments
        model: Keras model.
        mean: Tuple, containing mean per channel on ImageNet.
    """
    def __init__(self, model, mean):
        super(EfficientPosePreprocess, self).__init__()
        self.compute_resizing_shape = pr.ComputeResizingShape(
            model.input_shape[1])
        self.resize_image = pr.ResizeImageDynamic()
        self.preprocess = pr.SequentialProcessor([
            pr.SubtractMeanImage(mean),
            pr.PadImage(model.input_shape[1]),
            pr.CastImage(float),
            pr.ExpandDims(axis=0)])

    def call(self, image):
        resizing_shape, image_scale = self.compute_resizing_shape(image)
        preprocessed_image = self.resize_image(image, resizing_shape)
        preprocessed_image = self.preprocess(preprocessed_image)
        return preprocessed_image, image_scale


class EfficientPosePostprocess(Processor):
    """Postprocessing pipeline for EfficientPose.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating class names.
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        variances: List of float values.
        class_arg: Int, index of the class to be removed.
        num_pose_dims: Int, number of dimensions for pose.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 camera_matrix, regress_translation, compute_tx_ty_tz,
                 variances=[0.1, 0.1, 0.2, 0.2], class_arg=None,
                 num_pose_dims=3):
        super(EfficientPosePostprocess, self).__init__()
        self.num_pose_dims = num_pose_dims
        self.camera_matrix = camera_matrix
        self.postprocess_1 = pr.SequentialProcessor([
            pr.Squeeze(axis=None),
            pr.DecodeBoxes(model.prior_boxes, variances),
            pr.RemoveClass(class_names, class_arg)])
        self.scale_box = pr.ScaleBox()
        self.postprocess_2 = pr.SequentialProcessor([
            pr.NonMaximumSuppressionPerClass(nms_thresh),
            pr.MergeNMSBoxWithClass(),
            pr.FilterBoxes(class_names, score_thresh)])
        self.to_boxes2D = pr.ToBoxes2D(class_names)
        self.round_boxes = pr.RoundBoxes2D()
        self.denormalize = pr.DenormalizeBoxes2D()
        self.regress_translation = regress_translation
        self.compute_tx_ty_tz = compute_tx_ty_tz
        self.compute_common_rows = pr.ComputeCommonRowIndices()
        self.squeeze = pr.Squeeze(axis=0)
        self.transform_rotations = pr.Scale(np.pi)
        self.to_pose_6D = pr.ToPose6D(class_names)

    def call(self, image, model_output, image_scale):
        detections, transformations = model_output
        box_data = self.postprocess_1(detections)
        box_data = self.scale_box(box_data, 1 / image_scale)
        box_data_all = box_data
        box_data = self.postprocess_2(box_data)
        boxes2D = self.to_boxes2D(box_data)
        boxes2D = self.denormalize(image, boxes2D)
        boxes2D = self.round_boxes(boxes2D)
        rotations = transformations[:, :, :self.num_pose_dims]
        translations = transformations[:, :, self.num_pose_dims:]
        poses6D = []
        if len(boxes2D) > 0:
            selected_indices = self.compute_common_rows(box_data_all, box_data)
            rotations = self.squeeze(rotations)
            rotations = rotations[selected_indices]
            rotations = self.transform_rotations(rotations)
            translation_xy_Tz = self.regress_translation(translations)
            translation = self.compute_tx_ty_tz(
                translation_xy_Tz, self.camera_matrix, image_scale)
            translations = translation[selected_indices]
        poses6D = self.to_pose_6D(box_data, rotations, translations)
        return boxes2D, poses6D


class EstimateEfficientPose(Processor):
    """Object detection and pose estimation for EfficientPose models.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating class names.
        score_thresh: Float between [0, 1].
        nms_thresh: Float between [0, 1].
        camera_matrix: Array of shape `(3, 3)` camera matrix.
        object_sizes: Dict, dataset object sizes.
        preprocess: Callable, preprocessing pipeline.
        postprocess: Callable, postprocessing pipeline.
        variances: List of float values.
        show_boxes2D: Boolean. If ``True`` prediction
            are drawn in the returned image.
        show_poses6D: Boolean. If ``True`` estimated poses
            are drawn in the returned image.

    # Properties
        model: Keras model.
        class_names: List.
        score_thresh: Float.
        nms_thresh: Float.
        camera_matrix: Array.
        variances: List.
        class_to_sizes: Dict.
        colors: List.
        show_boxes2D: Bool.
        show_poses6D: Bool.
        preprocess: Callable.
        postprocess: Callable.
        draw_boxes2D: Callable.
        wrap: Callable.

    # Methods
        _build_draw_pose6D()
        call()
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 object_sizes, mean, camera_matrix, regress_translation,
                 compute_tx_ty_tz, preprocess=None, postprocess=None,
                 variances=[0.1, 0.1, 0.2, 0.2], show_boxes2D=False,
                 show_poses6D=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.camera_matrix = camera_matrix
        self.class_to_sizes = object_sizes
        self.mean = mean
        self.show_boxes2D = show_boxes2D
        self.show_poses6D = show_poses6D
        if preprocess is None:
            self.preprocess = EfficientPosePreprocess(model, mean)
        if postprocess is None:
            self.postprocess = EfficientPosePostprocess(
                model, class_names, score_thresh,
                nms_thresh, camera_matrix, regress_translation,
                compute_tx_ty_tz, class_arg=0)

        super(EstimateEfficientPose, self).__init__()
        self.draw_boxes2D = pr.DrawBoxes2D(self.class_names)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

    def _build_draw_pose6D(self, name_to_size, camera_parameter):
        name_to_draw = {}
        for name, object_size in name_to_size.items():
            draw = pr.DrawPose6D(object_size, camera_parameter)
            name_to_draw[name] = draw
        return name_to_draw

    def call(self, image):
        preprocessed_data = self.preprocess(image)
        preprocessed_image, image_scale = preprocessed_data
        outputs = self.model(preprocessed_image)
        detections, transformations = outputs
        outputs = detections, transformations
        boxes2D, poses6D = self.postprocess(preprocessed_image[0],
                                            outputs, image_scale)
        if self.show_boxes2D:
            image = self.draw_boxes2D(image, boxes2D)
        if self.show_poses6D:
            self.draw_pose6D = self._build_draw_pose6D(
                self.class_to_sizes, self.camera_matrix)
            for box2D, pose6D in zip(boxes2D, poses6D):
                image = self.draw_pose6D[box2D.class_name](image, pose6D)
        return self.wrap(image, boxes2D, poses6D)
