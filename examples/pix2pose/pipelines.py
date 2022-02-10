import numpy as np
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor, Pose6D
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz.pipelines import Pix2Points
from paz.backend.groups.quaternion import rotation_vector_to_quaternion
from paz.backend.keypoints import build_cube_points3D
from paz.backend.keypoints import denormalize_keypoints2D
from paz.backend.image.draw import draw_points2D
from paz.backend.keypoints import points3D_to_RGB
from paz.pipelines import SSD300FAT
from paz.models import UNET_VGG16


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


class Pix2Pose(Processor):
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
                 valid_class_names=['035_power_drill']):
        super(Pix2Pose, self).__init__()
        self.detect = detect
        self.estimate_pose = estimate_pose
        self.postprocess_boxes = SequentialProcessor(
            [pr.UnpackDictionary(['boxes2D']),
             pr.FilterClassBoxes2D(valid_class_names),
             pr.SquareBoxes2D(),
             pr.OffsetBoxes2D(offsets)])
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
                image = draw_points2D(image, set_points2D, colors)
            for pose6D in poses6D:
                image = self.draw_pose6D(image, pose6D)
        return self.wrap(image, boxes2D, poses6D)


class RGBMaskToPowerDrillPose6D(RGBMaskToPose6D):
    def __init__(self, camera, weights=None, epsilon=0.15,
                 resize=False, draw=True):
        model = UNET_VGG16(3, (128, 128, 3))
        model.load_weights('experiments/UNET-VGG16_RUN_00_08-02-2022_14-39-55/weights.hdf5')
        # segment.load_weights(weights)
        object_sizes = np.array([1840, 1870, 520])
        class_name = '035_power_drill'
        super(RGBMaskToPowerDrillPose6D, self).__init__(
            model, object_sizes, camera, epsilon, resize, class_name, draw)


class Pix2PosePowerDrill(Pix2Pose):
    def __init__(self, camera, score_thresh=0.50, epsilon=0.15,
                 offsets=[0.5, 0.5], draw=True):
        valid_class_names = ['035_power_drill']
        detect = SSD300FAT(score_thresh, draw=False)
        estimate_pose = RGBMaskToPowerDrillPose6D(camera, epsilon, draw=False)
        super(Pix2PosePowerDrill, self).__init__(
            detect, estimate_pose, offsets, draw, valid_class_names)
