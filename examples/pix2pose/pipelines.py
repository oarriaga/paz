# What other names are possible?
# TODO SingleInference, MultiInference, SingleClass, MultiClass
# SingleBox, MultiBox the problem is that SingleInference might
# not necessarily take a box
# TODO change append values to append_values_keys, append_values_list
# TODO verify that default offsets are provided for power drill
# TODO: Divide sizes by 10000 to obtain meters
# TODO change MultiInference for MultiInstance?

from paz import processors as pr
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz.backend.image import draw_points2D, draw_cube
from paz.backend.keypoints import points3D_to_RGB, denormalize_keypoints2D
from paz.backend.keypoints import project_to_image, build_cube_points3D
from paz.backend.groups import quaternion_to_rotation_matrix
from paz.pipelines.masks import Pix2Points
from paz.processors import Processor, SequentialProcessor
from paz.abstract.messages import Pose6D
from tensorflow.keras.utils import get_file
from paz.models import UNET_VGG16
from paz.pipelines import SSD300FAT
import numpy as np


class DomainRandomization(SequentialProcessor):
    """Performs domain randomization on a rendered image
    """
    def __init__(self, renderer, image_shape, image_paths, inputs_to_shape,
                 labels_to_shape, num_occlusions=1):
        super(DomainRandomization, self).__init__()
        H, W = image_shape[:2]
        self.add(pr.Render(renderer))
        self.add(pr.ControlMap(RandomizeRender(image_paths), [0, 1], [0]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [1], [1]))
        self.add(pr.SequenceWrapper({0: inputs_to_shape},
                                    {1: labels_to_shape}))


class PostprocessBoxes2D(SequentialProcessor):
    """Filters, squares and offsets 2D bounding boxes

    # Arguments
        valid_names: List of strings containing class names to keep.
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
    """
    def __init__(self, offsets, valid_names=None):
        super(PostprocessBoxes2D, self).__init__()
        if valid_names is not None:
            self.add(pr.FilterClassBoxes2D(valid_names))
        self.add(pr.SquareBoxes2D())
        self.add(pr.OffsetBoxes2D(offsets))


def append_values(dictionary, lists, keys):
    """Append dictionary values to lists

    # Arguments
        dictionary: dict
        lists: List of lists
        keys: Keys to dictionary values
    """
    if len(keys) != len(lists):
        assert ValueError('keys and lists must have same length')
    for key_arg, key in enumerate(keys):
        lists[key_arg].append(dictionary[key])
    return lists


def append_lists(intro_lists, outro_lists):
    """Appends multiple new values in intro lists to multiple outro lists

    # Arguments
        intro_lists: List of lists
        outro_lists: List of lists

    # Returns
        Lists with new values of intro lists
    """
    for intro_list, outro_list in zip(intro_lists, outro_lists):
        outro_list.append(intro_list)
    return outro_lists


class AppendValues(Processor):
    """Append dictionary values to lists

    # Arguments
        keys: Keys to dictionary values
    """
    def __init__(self, keys):
        super(AppendValues, self).__init__()
        self.keys = keys

    def call(self, dictionary, lists):
        return append_values(dictionary, lists, self.keys)


def draw_RGB_mask(image, points2D, points3D, object_sizes):
    """Draws RGB mask by transforming points3D to RGB space and putting in
        them in their 2D coordinates (points2D)

    # Arguments
        image: Array (H, W, 3).
        points2D: Array (num_points, 2)
        points3D: Array (num_points, 3)
        object_sizes: Array (x_size, y_size, z_size)

    # Returns
        Image array with drawn masks
    """
    color = points3D_to_RGB(points3D, object_sizes)
    image = draw_points2D(image, points2D, color)
    return image


class DrawRGBMask(Processor):
    """Draws RGB mask by transforming points3D to RGB space and putting in
        them in their 2D coordinates (points2D)

    # Arguments
        object_sizes: Array (x_size, y_size, z_size)
    """
    def __init__(self, object_sizes):
        super(DrawRGBMask, self).__init__()
        self.object_sizes = object_sizes

    def call(self, image, points2D, points3D):
        image = draw_RGB_mask(image, points2D, points3D, self.object_sizes)
        return image


def draw_RGB_masks(image, points2D, points3D, object_sizes):
    for instance_points2D, instance_points3D in zip(points2D, points3D):
        image = draw_RGB_mask(
            image, instance_points2D, instance_points3D, object_sizes)
    return image


class DrawRGBMasks(Processor):
    def __init__(self, object_sizes):
        super(DrawRGBMasks, self).__init__()
        self.object_sizes = object_sizes

    def call(self, image, points2D, points3D):
        return draw_RGB_masks(image, points2D, points3D, self.object_sizes)


def draw_pose6D(image, pose6D, points3D, intrinsics, thickness):
    """Draws cube in image by projecting points3D with intrinsics and pose6D.

    # Arguments
        image: Array (H, W).
        pose6D: paz.abstract.Pose6D instance.
        intrinsics: Array (3, 3). Camera intrinsics for projecting
            3D rays into 2D image.
        points3D: Array (num_points, 3).
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    quaternion, translation = pose6D.quaternion, pose6D.translation
    rotation = quaternion_to_rotation_matrix(quaternion)
    points2D = project_to_image(rotation, translation, points3D, intrinsics)
    image = draw_cube(image, points2D.astype(np.int32), thickness=thickness)
    return image


class DrawPoses6D(Processor):
    """Draws multiple cubes in image by projecting points3D.

    # Arguments
        object_sizes: Array (3) indicating (x, y, z) sizes of object.
        camera_intrinsics: Array (3, 3).
            Camera intrinsics for projecting 3D rays into 2D image.
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    def __init__(self, object_sizes, camera_intrinsics, thickness=2):
        self.points3D = build_cube_points3D(*object_sizes)
        self.intrinsics = camera_intrinsics
        self.thickness = thickness

    def call(self, image, poses6D):
        if poses6D is None:
            return image
        if not isinstance(poses6D, list):
            raise ValueError('Poses6D must be a list of Pose6D messages')
        for pose6D in poses6D:
            image = draw_pose6D(
                image, pose6D, self.points3D, self.intrinsics, self.thickness)
        return image


class DrawPose6D(Processor):
    """Draws a single cube in image by projecting points3D.

    # Arguments
        object_sizes: Array (3) indicating (x, y, z) sizes of object.
        camera_intrinsics: Array (3, 3).
            Camera intrinsics for projecting 3D rays into 2D image.
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    def __init__(self, object_sizes, camera_intrinsics, thickness=2):
        self.points3D = build_cube_points3D(*object_sizes)
        self.intrinsics = camera_intrinsics
        self.thickness = thickness

    def call(self, image, pose6D):
        if pose6D is None:
            return image
        image = draw_pose6D(
            image, pose6D, self.points3D, self.intrinsics, self.thickness)
        return image


def translate_points2D_origin(points2D, coordinates):
    """Translates points2D to a different origin

    # Arguments
        points2D: Array (num_points, 2)
        coordinates: Array (4) containing (x_min, y_min, x_max, y_max)

    # Returns
        Translated points2D array (num_points, 2)
    """
    x_min, y_min, x_max, y_max = coordinates
    points2D[:, 0] = points2D[:, 0] + x_min
    points2D[:, 1] = points2D[:, 1] + y_min
    return points2D


class SingleInferencePIX2POSE6D(Processor):
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
        super(SingleInferencePIX2POSE6D, self).__init__()
        self.camera = camera
        self.pix2points = Pix2Points(model, object_sizes, epsilon, resize)
        self.solvePnP = pr.SolveChangingObjectPnPRANSAC(self.camera.intrinsics)
        self.draw_pose6D = DrawPose6D(object_sizes, self.camera.intrinsics)
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


class MultiInferencePIX2POSE(Processor):
    """Predicts poses6D of multiple instances the same object from a single image.

    # Arguments
        estimate_pose: Function that takes as input an image and outputs a
            dictionary with points2D, points3D and pose6D messages e.g
            SingleInferencePIX2POSE6D
        offsets: List of length two containing floats e.g. (x_scale, y_scale)
        camera: PAZ Camera with intrinsic matrix.
        draw: Boolean. If True drawing functions are applied to output image.

    # Returns
        Dictionary with inferred boxes2D, poses6D and image.
    """
    def __init__(self, estimate_pose, offsets, camera=None, draw=True):
        super(MultiInferencePIX2POSE, self).__init__()
        self.draw = draw
        self.estimate_pose = estimate_pose
        self.object_sizes = self.estimate_pose.object_sizes
        self.camera = self.estimate_pose.camera if camera is None else camera
        valid_names = [self.estimate_pose.class_name]
        self.postprocess_boxes = PostprocessBoxes2D(offsets, valid_names)

        self.append_values = AppendValues(['pose6D', 'points2D', 'points3D'])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.draw_RGBmask = DrawRGBMasks(self.object_sizes)
        self.draw_boxes2D = pr.DrawBoxes2D(valid_names, colors=[[0, 255, 0]])
        self.draw_poses6D = DrawPoses6D(self.object_sizes, camera.intrinsics)
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


class PIX2SinglePowerDrillPose6D(SingleInferencePIX2POSE6D):
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
        super(PIX2SinglePowerDrillPose6D, self).__init__(
            model, object_sizes, camera, epsilon, resize, class_name, draw)


class PIX2MultiPowerDrillPoses6D(MultiInferencePIX2POSE):
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
        estimate_pose = PIX2SinglePowerDrillPose6D(
            camera, epsilon, resize, draw=False)
        super(PIX2MultiPowerDrillPoses6D, self).__init__(
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
        self.estimate_pose = PIX2MultiPowerDrillPoses6D(
            camera, offsets, epsilon, resize, draw)

    def call(self, image):
        return self.estimate_pose(image, self.detect(image)['boxes2D'])


class MultiInferenceMultiClassPIX2POSE(Processor):
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
        super(MultiInferenceMultiClassPIX2POSE, self).__init__()
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
            draw = DrawPose6D(object_sizes, camera.intrinsics)
            name_to_draw[name] = draw
        return name_to_draw

    def _build_draw_RGBmask(self, name_to_size):
        name_to_draw = {}
        for name, object_sizes in name_to_size.items():
            draw = DrawRGBMask(object_sizes)
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


class PIX2YCBTools6D(MultiInferenceMultiClassPIX2POSE):
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
        UNET_power_drill = UNET_VGG16(3, (128, 128, 3))
        URL = ('https://github.com/oarriaga/altamira-data/'
               'releases/download/v0.13/')
        name = 'UNET-VGG16_POWERDRILL_weights.hdf5'
        weights_path = get_file(name, URL + name, cache_subdir='paz/models')
        UNET_power_drill.load_weights(weights_path)

        UNET_large_clamp = UNET_VGG16(3, (128, 128, 3))
        UNET_large_clamp.load_weights(
            'experiments/UNET-VGG16_RUN_00_07-04-2022_13-28-04/'
            'model_weights.hdf5')

        UNET_scissors = UNET_VGG16(3, (128, 128, 3))
        UNET_scissors.load_weights(
            'experiments/UNET-VGG16_RUN_00_04-04-2022_12-29-44/'
            'model_weights.hdf5')

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
