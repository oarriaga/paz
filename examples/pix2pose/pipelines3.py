from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz.abstract.messages import Pose6D
from paz import processors as pr
from processors import (
    GetNonZeroArguments, GetNonZeroValues, ArgumentsToImagePoints2D,
    ImageToNormalizedDeviceCoordinates, Scale, SolveChangingObjectPnPRANSAC,
    ReplaceLowerThanThreshold)
from backend import build_cube_points3D
from processors import UnwrapDictionary
from processors import NormalizePoints2D
from backend import denormalize_points2D
from backend import draw_poses6D
from backend import draw_masks
from backend import draw_mask
from backend import normalize_points2D
from backend import draw_pose6D
from paz.backend.quaternion import rotation_vector_to_quaternion
from paz.backend.image import resize_image, show_image
from pipelines import SolveChangingObjectPnP
from pipelines import RGBMaskToImagePoints2D, RGBMaskToObjectPoints3D, PredictRGBMask


class Pix2Points(pr.Processor):
    def __init__(self, model, object_sizes, epsilon=0.15, resize=True):
        self.object_sizes = object_sizes
        self.predict_RGBMask = PredictRGBMask(model, epsilon)
        self.mask_to_points3D = RGBMaskToObjectPoints3D(self.object_sizes)
        self.mask_to_points2D = RGBMaskToImagePoints2D(model.output_shape[1:3])
        self.resize = resize
        self.wrap = pr.WrapOutput(['points2D', 'points3D', 'RGB_mask'])

    def call(self, image):
        RGB_mask = self.predict_RGBMask(image)
        H, W, num_channels = image.shape
        if self.resize:
            RGB_mask = resize_image(RGB_mask, (W, H))
        points3D = self.mask_to_points3D(RGB_mask)
        points2D = self.mask_to_points2D(RGB_mask)
        points2D = normalize_points2D(points2D, H, W)
        return self.wrap(points2D, points3D, RGB_mask)


class Pix2Pose(pr.Processor):
    def __init__(self, model, object_sizes, camera,
                 epsilon=0.15, class_name=None, draw=True):

        self.pix2points = Pix2Points(model, object_sizes, epsilon, True)
        self.predict_pose = SolveChangingObjectPnP(camera.intrinsics)
        self.class_name = str(class_name) if class_name is None else class_name
        self.object_sizes = object_sizes
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.camera = camera
        self.draw = draw

    def call(self, image, box2D=None):
        results = self.pix2points(image)
        points2D, points3D = results['points2D'], results['points3D']
        H, W, num_channels = image.shape
        points2D = denormalize_points2D(points2D, H, W)
        if box2D is not None:
            points2D = self.change_coordinates(points2D, box2D)
            self.class_name = box2D.class_name

        min_num_points = len(points3D) > self.predict_pose.MIN_REQUIRED_POINTS
        if min_num_points:
            pose_results = self.predict_pose(points3D, points2D)
            success, rotation, translation = pose_results
        if success and min_num_points:
            quaternion = rotation_vector_to_quaternion(rotation)
            pose6D = Pose6D(quaternion, translation, self.class_name)
        else:
            pose6D = None
        if self.draw:
            topic = 'image_crop' if box2D is not None else 'image'
            image = draw_mask(image, points2D, points3D, self.object_sizes)
            image = draw_pose6D(image, pose6D, self.cube_points3D,
                                self.camera.intrinsics)
            results[topic] = image
        results['points2D'], results['pose6D'] = points2D, pose6D
        return results


class EstimatePoseMasks(Processor):
    def __init__(self, detect, estimate_pose, offsets, draw=True):
        """Pose estimation pipeline using keypoints.
        """
        super(EstimatePoseMasks, self).__init__()
        self.detect = detect
        self.estimate_pose = estimate_pose
        self.postprocess_boxes = SequentialProcessor(
            [pr.UnpackDictionary(['boxes2D']),
             pr.FilterClassBoxes2D(['035_power_drill']),
             pr.SquareBoxes2D(),
             pr.OffsetBoxes2D(offsets)])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.unwrap = UnwrapDictionary(['pose6D', 'points2D', 'points3D'])
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.object_sizes = self.estimate_pose.object_sizes
        self.cube_points3D = build_cube_points3D(*self.object_sizes)
        self.draw = draw

    def call(self, image):
        boxes2D = self.postprocess_boxes(self.detect(image))
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points = [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            results = self.estimate_pose(crop, box2D)
            pose6D, points2D, points3D = self.unwrap(results)
            poses6D.append(pose6D), points.append([points2D, points3D])
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
            image = draw_masks(image, points, self.object_sizes)
            image = draw_poses6D(image, poses6D, self.cube_points3D,
                                 self.estimate_pose.camera.intrinsics)
        return self.wrap(image, boxes2D, poses6D)
