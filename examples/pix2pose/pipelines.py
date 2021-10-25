import numpy as np
from paz.abstract import SequentialProcessor
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz import processors as pr
from processors import (
    GetNonZeroArguments, GetNonZeroValues, ArgumentsToImagePoints2D,
    ImageToClosedOneBall, Scale, SolveChangingObjectPnPRANSAC,
    RotationVectorToRotationMatrix, ReplaceLowerThanThreshold)
from backend import build_cube_points3D, project_to_image, draw_cube
from processors import CropImage
from paz.backend.image import show_image


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
        self.add(CropImage())
        self.add(pr.ResizeImage((128, 128)))
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


class Pix2Pose(pr.Processor):
    def __init__(self, model, object_sizes, camera, epsilon=0.15):
        self.camera = camera
        self.object_sizes = object_sizes
        self.predict_RGBMask = PredictRGBMask(model, epsilon)
        self.RGBMask_to_object_points3D = RGBMaskToObjectPoints3D(
            self.object_sizes)
        self.RGBMask_to_image_points2D = RGBMaskToImagePoints2D()
        self.predict_pose = SolveChangingObjectPnPRANSAC(camera.intrinsics)
        self.vector_to_matrix = RotationVectorToRotationMatrix()

    def call(self, image):
        show_image(image, wait=False)
        RGBMask = self.predict_RGBMask(image)
        print(RGBMask.shape)
        return {'image': RGBMask}
        points3D = self.RGBMask_to_object_points3D(RGBMask)
        points2D = self.RGBMask_to_image_points2D(RGBMask)
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
