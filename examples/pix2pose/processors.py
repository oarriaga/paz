import numpy as np
from paz.abstract import Processor
from paz.backend.keypoints import project_points3D
from paz.backend.image import draw_cube
from paz.backend.quaternion import rotation_vector_to_quaternion

from backend import build_cube_points3D
from backend import replace_lower_than_threshold
from backend import arguments_to_image_points2D
from backend import solve_PnP_RANSAC
from backend import rotation_vector_to_rotation_matrix
from backend import translate_points2D
from backend import normalize_points2D


class ImageToClosedOneBall(Processor):
    """Map image value from [0, 255] -> [-1, 1].
    """
    def __init__(self):
        super(ImageToClosedOneBall, self).__init__()

    def call(self, image):
        return (image / 127.5) - 1.0


class ClosedOneBallToImage(Processor):
    """Map normalized value from [-1, 1] -> [0, 255].
    """
    def __init__(self):
        super(ClosedOneBallToImage, self).__init__()

    def call(self, image):
        return (image + 1.0) * 127.5


class DrawBoxes3D(Processor):
    def __init__(self, camera, class_to_dimensions, thickness=1):
        """Draw boxes 3D of multiple objects

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera''.
            class_to_dimensions: Dictionary that has as keys the
                class names and as value a list [model_height, model_width]
            thickness: Int. Thickness of 3D box
        """
        super(DrawBoxes3D, self).__init__()
        self.camera = camera
        self.class_to_dimensions = class_to_dimensions
        self.class_to_points = self._build_points(self.class_to_dimensions)
        self.thickness = thickness

    def _build_points(self, class_to_dimensions):
        class_to_cube3D = {}
        print(class_to_dimensions)
        for class_name, dimensions in class_to_dimensions.items():
            width, height, depth = dimensions
            cube_points3D = build_cube_points3D(width, height, depth)
            class_to_cube3D[class_name] = cube_points3D
        return class_to_cube3D

    def call(self, image, pose6D):
        points3D = self.class_to_points[pose6D.class_name]
        points2D = project_points3D(points3D, pose6D, self.camera)
        points2D = points2D.astype(np.int32)
        # points2D = np.squeeze(points2D)
        # return points2D
        draw_cube(image, points2D, thickness=self.thickness)
        return image


class ReplaceLowerThanThreshold(Processor):
    def __init__(self, threshold=1e-8, replacement=0.0):
        super(ReplaceLowerThanThreshold, self).__init__()
        self.threshold = threshold
        self.replacement = replacement

    def call(self, image):
        return replace_lower_than_threshold(
            image, self.threshold, self.replacement)


class GetNonZeroValues(Processor):
    def __init__(self):
        super(GetNonZeroValues, self).__init__()

    def call(self, array):
        channel_wise_sum = np.sum(array, axis=2)
        non_zero_arguments = np.nonzero(channel_wise_sum)
        return array[non_zero_arguments]


class GetNonZeroArguments(Processor):
    def __init__(self):
        super(GetNonZeroArguments, self).__init__()

    def call(self, array):
        channel_wise_sum = np.sum(array, axis=2)
        non_zero_rows, non_zero_columns = np.nonzero(channel_wise_sum)
        return non_zero_rows, non_zero_columns


class ArgumentsToImagePoints2D(Processor):
    def __init__(self):
        super(ArgumentsToImagePoints2D, self).__init__()

    def call(self, row_args, col_args):
        image_points2D = arguments_to_image_points2D(row_args, col_args)
        return image_points2D


class Scale(Processor):
    def __init__(self, object_sizes):
        super(Scale, self).__init__()
        self.object_sizes = object_sizes

    def call(self, values):
        return self.object_sizes * values


class SolveChangingObjectPnPRANSAC(Processor):
    def __init__(self, camera_intrinsics, inlier_thresh=5, num_iterations=100):
        super(SolveChangingObjectPnPRANSAC, self).__init__()
        self.camera_intrinsics = camera_intrinsics
        self.inlier_thresh = inlier_thresh
        self.num_iterations = num_iterations

    def call(self, object_points3D, image_points2D):
        rotation_vector, translation = solve_PnP_RANSAC(
            object_points3D, image_points2D, self.camera_intrinsics,
            self.inlier_thresh, self.num_iterations)
        return rotation_vector, translation


class RotationVectorToRotationMatrix(Processor):
    def __init__(self):
        super(RotationVectorToRotationMatrix, self).__init__()

    def call(self, rotation_vector):
        return rotation_vector_to_rotation_matrix(rotation_vector)


class CropImage(Processor):
    def __init__(self):
        super(CropImage, self).__init__()

    def call(self, image):
        return image[:128, :128, :]


class UnwrapDictionary(Processor):
    def __init__(self, keys):
        super(UnwrapDictionary, self).__init__()
        self.keys = keys

    def call(self, dictionary):
        return [dictionary[key] for key in self.keys]


class ToAffineMatrix(Processor):
    def __init__(self):
        super(ToAffineMatrix, self).__init__()

    def call(self, rotation_matrix, translation):
        if len(translation) != 3:
            raise ValueError('Translation should be of lenght 3')
        if rotation_matrix.shape != (3, 3):
            raise ValueError('Rotation matrix should be of shape (3, 3)')
        translation = translation.reshape(3, 1)
        affine_matrix = np.concatenate([rotation_matrix, translation], axis=1)
        affine_row = np.array([[0.0, 0.0, 0.0, 1.0]])
        affine_matrix = np.concatenate([affine_matrix, affine_row], axis=0)
        print(affine_matrix.shape)
        return affine_matrix


class RotationVectorToQuaternion(Processor):
    def __init__(self):
        super(RotationVectorToQuaternion, self).__init__()

    def call(self, rotation_vector):
        if rotation_vector is None:
            return None
        quaternion = rotation_vector_to_quaternion(rotation_vector)
        return quaternion


class TranslatePoints2D(Processor):
    def __init__(self):
        super(TranslatePoints2D, self).__init__()

    def call(points2D, image):
        height, width = image.shape[:2]
        translated_points2D = translate_points2D(points2D, (height, width))
        return translated_points2D


class FlipYAxisPoints2D(Processor):
    def __init__(self):
        super(FlipYAxisPoints2D, self).__init__()

    def call(self, points2D, image):
        height = image.shape[0]
        translate_points2D(points2D, (0, height))


class NormalizePoints2D(Processor):
    def __init__(self, image_shape):
        self.height, self.width = image_shape[:2]

    def call(self, points2D):
        points2D = normalize_points2D(points2D, self.height, self.width)
        return points2D
