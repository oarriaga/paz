from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz.abstract.messages import Pose6D
from paz import processors as pr
from processors import (
    GetNonZeroArguments, GetNonZeroValues, ArgumentsToImagePoints2D,
    ImageToClosedOneBall, Scale, SolveChangingObjectPnPRANSAC,
    ReplaceLowerThanThreshold)
from backend import build_cube_points3D
from processors import UnwrapDictionary
from processors import NormalizePoints2D
from backend import denormalize_points2D
from backend import draw_poses6D
from backend import draw_masks
from backend import normalize_points2D
from paz.backend.quaternion import rotation_vector_to_quaternion
from paz.backend.image import resize_image, show_image


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
        # self.add(pr.ControlMap(ImageToClosedOneBall(), [1], [1]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [1], [1]))
        """
        self.add(pr.SequenceWrapper({0: {'input_1': [H, W, 3]}},
                                    {1: {'masks': [H, W, 4]}}))
        """
        self.add(pr.SequenceWrapper({0: inputs_to_shape},
                                    {1: labels_to_shape}))


class PredictRGBMask(SequentialProcessor):
    def __init__(self, model, epsilon=0.15):
        super(PredictRGBMask, self).__init__()
        self.add(pr.ResizeImage(model.input_shape[1:3]))
        self.add(pr.NormalizeImage())
        self.add(pr.ExpandDims(0))
        self.add(pr.Predict(model))
        self.add(pr.Squeeze(0))
        self.add(ReplaceLowerThanThreshold(epsilon))
        self.add(pr.DenormalizeImage())
        self.add(pr.CastImage('uint8'))


class RGBMaskToObjectPoints3D(SequentialProcessor):
    def __init__(self, object_sizes):
        super(RGBMaskToObjectPoints3D, self).__init__()
        self.add(GetNonZeroValues())
        self.add(ImageToClosedOneBall())
        self.add(Scale(object_sizes / 2.0))


class RGBMaskToImagePoints2D(SequentialProcessor):
    def __init__(self, output_shape):
        super(RGBMaskToImagePoints2D, self).__init__()
        self.add(GetNonZeroArguments())
        self.add(ArgumentsToImagePoints2D())
        # self.add(NormalizePoints2D(output_shape))


class SolveChangingObjectPnP(SequentialProcessor):
    def __init__(self, camera_intrinsics, inlier_thresh=5, num_iterations=100):
        super(SolveChangingObjectPnP, self).__init__()
        self.MINIMUM_REQUIRED_POINTS = 4
        self.add(SolveChangingObjectPnPRANSAC(
            camera_intrinsics, inlier_thresh, num_iterations))


class Pix2Pose(pr.Processor):
    def __init__(self, model, object_sizes, epsilon=0.15, with_resize=True):
        self.object_sizes = object_sizes
        self.predict_RGBMask = PredictRGBMask(model, epsilon)
        self.mask_to_points3D = RGBMaskToObjectPoints3D(self.object_sizes)
        self.mask_to_points2D = RGBMaskToImagePoints2D(model.output_shape[1:3])
        self.wrap = pr.WrapOutput(['points3D', 'points2D', 'RGB_mask'])
        self.with_resize = with_resize

    def call(self, image):
        RGB_mask = self.predict_RGBMask(image)
        if self.with_resize:
            RGB_mask = resize_image(RGB_mask, image.shape[:2][::-1])
        points3D = self.mask_to_points3D(RGB_mask)
        points2D = self.mask_to_points2D(RGB_mask)
        points2D = normalize_points2D(points2D, *image.shape[:2][::-1])
        return self.wrap(points3D, points2D, RGB_mask)


class EstimatePoseMasks(Processor):
    def __init__(self, detect, estimate_keypoints, camera, offsets, draw=True):
        """Pose estimation pipeline using keypoints.
        """
        super(EstimatePoseMasks, self).__init__()
        self.detect = detect
        self.estimate_keypoints = estimate_keypoints
        self.camera = camera
        self.draw = draw
        self.postprocess_boxes = SequentialProcessor(
            [pr.UnpackDictionary(['boxes2D']),
             pr.FilterClassBoxes2D(['035_power_drill']),
             pr.SquareBoxes2D(),
             pr.OffsetBoxes2D(offsets)])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.predict_pose = SolveChangingObjectPnP(camera.intrinsics)
        self.unwrap = UnwrapDictionary(['points2D', 'points3D'])
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.object_sizes = self.estimate_keypoints.object_sizes
        self.cube_points3D = build_cube_points3D(*self.object_sizes)

    def call(self, image):
        boxes2D = self.postprocess_boxes(self.detect(image))
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points = [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            points2D, points3D = self.unwrap(self.estimate_keypoints(crop))
            points2D = denormalize_points2D(points2D, *crop.shape[0:2])
            points2D = self.change_coordinates(points2D, box2D)
            if len(points3D) < self.predict_pose.MINIMUM_REQUIRED_POINTS:
                continue
            success, rotation, translation = self.predict_pose(
                points3D, points2D)
            if success is False:
                continue
            print('ROTATION', rotation.shape)
            quaternion = rotation_vector_to_quaternion(rotation)
            print('QUATERNION', quaternion.shape)
            pose6D = Pose6D(quaternion, translation, box2D.class_name)
            poses6D.append(pose6D), points.append([points2D, points3D])
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            image = draw_masks(image, points, self.object_sizes)
            image = draw_poses6D(image, poses6D, self.cube_points3D,
                                 self.camera.intrinsics)
        return self.wrap(image, boxes2D, poses6D)
