import numpy as np
from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz.abstract.messages import Pose6D
from paz import processors as pr
from processors import (
    GetNonZeroArguments, GetNonZeroValues, ArgumentsToImagePoints2D,
    ImageToClosedOneBall, Scale, SolveChangingObjectPnPRANSAC,
    ReplaceLowerThanThreshold)
from backend import (build_cube_points3D, project_to_image, draw_cube,
                     draw_keypoints)
from processors import UnwrapDictionary, RotationVectorToQuaternion
# from paz.backend.image import show_image
from backend import quaternion_to_rotation_matrix, draw_maski
from backend import normalize_points2D, flip_y_axis
from backend import denormalize_points2D


class DomainRandomization(SequentialProcessor):
    """Performs domain randomization on a rendered image
    """
    def __init__(self, renderer, image_shape, image_paths, num_occlusions=1):
        super(DomainRandomization, self).__init__()
        H, W = image_shape[:2]
        self.add(pr.Render(renderer))
        self.add(pr.ControlMap(RandomizeRender(image_paths), [0, 1], [0]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        # self.add(pr.ControlMap(ImageToClosedOneBall(), [1], [1]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [1], [1]))
        self.add(pr.SequenceWrapper({0: {'input_1': [H, W, 3]}},
                                    {1: {'masks': [H, W, 4]}}))


class PredictRGBMask(SequentialProcessor):
    def __init__(self, model, epsilon=0.15):
        super(PredictRGBMask, self).__init__()
        # self.add(pr.ResizeImage((128, 128)))
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
    def __init__(self):
        super(RGBMaskToImagePoints2D, self).__init__()
        self.add(GetNonZeroArguments())
        self.add(ArgumentsToImagePoints2D())


class SolveChangingObjectPnP(SequentialProcessor):
    def __init__(self, camera_intrinsics):
        super(SolveChangingObjectPnP, self).__init__()
        self.add(SolveChangingObjectPnPRANSAC(camera_intrinsics))
        # self.add(pr.ControlMap(RotationVectorToRotationMatrix()))
        self.add(pr.ControlMap(RotationVectorToQuaternion()))
        self.add(pr.ControlMap(pr.Squeeze(1), [1], [1]))
        # self.add(ToPose6D())
        # self.add(ToAffineMatrix())


class Pix2Pose(pr.Processor):
    def __init__(self, model, object_sizes, epsilon=0.15):
        self.object_sizes = object_sizes
        H, W = model.input_shape[1:3]
        self.resize = pr.ResizeImage((W, H))
        self.predict_RGBMask = PredictRGBMask(model, epsilon)
        self.RGBMask_to_points3D = RGBMaskToObjectPoints3D(self.object_sizes)
        self.RGBMask_to_points2D = RGBMaskToImagePoints2D()
        self.wrap = pr.WrapOutput(['points3D', 'points2D', 'RGB_mask'])

    def call(self, image):
        image = self.resize(image)
        RGB_mask = self.predict_RGBMask(image)
        points3D = self.RGBMask_to_points3D(RGB_mask)
        points2D = self.RGBMask_to_points2D(RGB_mask)
        return self.wrap(points3D, points2D, RGB_mask)
        """
        rotation_vector, translation = self.predict_pose(points3D, points2D)
        rotation_matrix = self.vector_to_matrix(rotation_vector)
        translation = np.squeeze(translation, 1)
        points3D = build_cube_points3D(*self.object_sizes)
        points2D = project_to_image(
            rotation_matrix, translation, points3D, self.camera.intrinsics)
        points2D = points2D.astype(np.int32)
        image = draw_cube(image.astype(float), points2D)
        image = image.astype('uint8')
        return {'image', image}
        """


class EstimatePoseMasks(Processor):
    def __init__(self, detect, estimate_keypoints, camera, offsets,
                 class_to_dimensions, radius=3, thickness=1):
        """Pose estimation pipeline using keypoints.
        """
        super(EstimatePoseMasks, self).__init__()
        self.detect = detect
        self.camera = camera
        self.estimate_keypoints = estimate_keypoints
        self.square = SequentialProcessor(
            [pr.SquareBoxes2D(), pr.OffsetBoxes2D(offsets)])
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.predict_pose = SolveChangingObjectPnP(camera.intrinsics)
        self.unwrap = UnwrapDictionary(['points2D', 'points3D'])
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])
        self.draw_boxes2D = pr.DrawBoxes2D(detect.class_names)
        self.cube_points3D = build_cube_points3D(0.2, 0.2, 0.07)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, points = [], []
        for crop, box2D in zip(cropped_images, boxes2D):
            if box2D.class_name != '035_power_drill':
                continue
            points2D, points3D = self.unwrap(self.estimate_keypoints(crop))

            points2D = normalize_points2D(points2D, 128.0, 128.0)
            crop_H, crop_W = crop.shape[:2]
            points2D = denormalize_points2D(points2D, crop_H, crop_W)
            points2D = self.change_coordinates(points2D, box2D)

            quaternion, translation = self.predict_pose(points3D, points2D)
            pose6D = Pose6D(quaternion, translation, box2D.class_name)

            poses6D.append(pose6D), points.append([points2D, points3D])

        # draw boxes
        new_boxes2D = []
        for box2D in boxes2D:
            if box2D.class_name == '035_power_drill':
                new_boxes2D.append(box2D)
        image = self.draw_boxes2D(image, new_boxes2D)

        # draw masks
        for points2D, points3D in points:
            object_sizes = np.array([0.184, 0.187, 0.052])
            colors = points3D / (object_sizes / 2.0)
            colors = (colors + 1.0) * 127.5
            colors = colors.astype('int')
            draw_maski(image, points2D, colors)

        # draw cubes
        image = image.astype(float)
        for pose6D in poses6D:
            rotation = quaternion_to_rotation_matrix(pose6D.quaternion)
            rotation = np.squeeze(rotation, axis=2)
            cube_points2D = project_to_image(
                rotation,
                pose6D.translation,
                self.cube_points3D,
                self.camera.intrinsics)
            cube_points2D = cube_points2D.astype(np.int32)
            image = draw_cube(image, cube_points2D)
        image = image.astype('uint8')

        return self.wrap(image, boxes2D, poses6D)
