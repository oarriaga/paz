from warnings import warn

import numpy as np

from ..abstract import Processor
from ..backend.keypoints import translate_keypoints
from ..backend.keypoints import arguments_to_image_points2D
from ..backend.keypoints import normalize_keypoints2D
from ..backend.keypoints import denormalize_keypoints2D
from ..backend.keypoints import normalize_keypoints
from ..backend.keypoints import denormalize_keypoints
from ..backend.keypoints import compute_orientation_vector
from ..backend.image import get_scaling_factor


class ProjectKeypoints(Processor):
    """Projects homogenous keypoints (4D) in the camera coordinates system into
        image coordinates using a projective transformation.

    # Arguments
        projector: Instance of ''paz.models.Project''.
        keypoints: Numpy array of shape ''(num_keypoints, 3)''
    """
    def __init__(self, projector, keypoints):
        self.projector = projector
        self.keypoints = keypoints
        super(ProjectKeypoints, self).__init__()

    def call(self, world_to_camera):
        keypoints = np.matmul(self.keypoints, world_to_camera.T)
        keypoints = np.expand_dims(keypoints, 0)
        keypoints = self.projector.project(keypoints)[0]
        return keypoints


class NormalizeKeypoints2D(Processor):
    """Transform keypoints in image-size coordinates to normalized coordinates.

    # Arguments
        image_size: List of two ints indicating ''(height, width)''
    """
    def __init__(self, image_size):
        self.image_size = image_size
        super(NormalizeKeypoints2D, self).__init__()

    def call(self, keypoints):
        height, width = self.image_size[0:2]
        keypoints = normalize_keypoints2D(keypoints, height, width)
        return keypoints


class DenormalizeKeypoints2D(Processor):
    """Transform normalized keypoints coordinates into image-size coordinates.

    # Arguments
        image_size: List of two floats having height and width of image.
    """
    def __init__(self):
        super(DenormalizeKeypoints2D, self).__init__()

    def call(self, keypoints, image):
        height, width = image.shape[0:2]
        keypoints = denormalize_keypoints2D(keypoints, height, width)
        return keypoints


class NormalizeKeypoints(Processor):
    """Transform keypoints in image-size coordinates to normalized coordinates.

    # Arguments
        image_size: List of two ints indicating ''(height, width)''
    """
    def __init__(self, image_size):
        self.image_size = image_size
        warn('DEPRECATED please use normalize_keypoints2D')
        super(NormalizeKeypoints, self).__init__()

    def call(self, keypoints):
        height, width = self.image_size[0:2]
        keypoints = normalize_keypoints(keypoints, height, width)
        return keypoints


class DenormalizeKeypoints(Processor):
    """Transform normalized keypoints coordinates into image-size coordinates.

    # Arguments
        image_size: List of two floats having height and width of image.
    """
    def __init__(self):
        warn('DEPRECATED please use denormalize_keypoints2D')
        super(DenormalizeKeypoints, self).__init__()

    def call(self, keypoints, image):
        height, width = image.shape[0:2]
        keypoints = denormalize_keypoints(keypoints, height, width)
        return keypoints


class RemoveKeypointsDepth(Processor):
    """Removes Z component from keypoints.
    """
    def __init__(self):
        super(RemoveKeypointsDepth, self).__init__()

    def call(self, keypoints):
        return keypoints[:, :2]


class PartitionKeypoints(Processor):
    """Partitions keypoints from shape [num_keypoints, 2] into a list of the form
        ((2), (2), ....) and length equal to num_of_keypoints.
    """
    def __init__(self):
        super(PartitionKeypoints, self).__init__()

    def call(self, keypoints):
        keypoints = np.vsplit(keypoints, len(keypoints))
        keypoints = [np.squeeze(keypoint) for keypoint in keypoints]
        partioned_keypoints = []
        for keypoint_arg, keypoint in enumerate(keypoints):
            partioned_keypoints.append(keypoint)
        return np.asarray(partioned_keypoints)


class ChangeKeypointsCoordinateSystem(Processor):
    """Changes ``keypoints`` 2D coordinate system using ``box2D`` coordinates
        to locate the new origin at the openCV image origin (top-left).
    """
    def __init__(self):
        super(ChangeKeypointsCoordinateSystem, self).__init__()

    def call(self, keypoints, box2D):
        x_min, y_min, x_max, y_max = box2D.coordinates
        keypoints[:, 0] = keypoints[:, 0] + x_min
        keypoints[:, 1] = keypoints[:, 1] + y_min
        return keypoints


class TranslateKeypoints(Processor):
    """Applies a translation to keypoints.
    The translation is a list of length two indicating the x, y values.
    """
    def __init__(self):
        super(TranslateKeypoints, self).__init__()

    def call(self, keypoints, translation):
        return translate_keypoints(keypoints, translation)


class ArgumentsToImageKeypoints2D(Processor):
    """Convert array arguments into UV coordinates.

              Image plane

           (0,0)-------->  (U)
             |
             |
             |
             v

            (V)

    # Arguments
        row_args: Array (num_rows).
        col_args: Array (num_cols).

    # Returns
        Array (num_cols, num_rows) representing points2D in UV space.

    # Notes
        Arguments are row args (V) and col args (U). Image points are in UV
            coordinates; thus, we concatenate them in that order
            i.e. [col_args, row_args]
    """
    def __init__(self):
        super(ArgumentsToImageKeypoints2D, self).__init__()

    def call(self, row_args, col_args):
        image_points2D = arguments_to_image_points2D(row_args, col_args)
        return image_points2D


class ScaleKeypoints(Processor):
    """Scale keypoints to input image shape.

    # Arguments
        keypoints: Array. Detected keypoints by the model
        image: Array. Input image.

    # Returns
        Scaled keypoints: Array. keypoints scaled to input image shape.
    """
    def __init__(self, scale=1, shape=(128, 128)):
        super(ScaleKeypoints, self).__init__()
        self.scale = scale
        self.shape = shape

    def call(self, keypoints, image):
        scale = get_scaling_factor(image, self.scale, self.shape)
        scaled_keypoints = keypoints * scale
        return np.array(scaled_keypoints, dtype=np.uint)


class ComputeOrientationVector(Processor):
    """Calculate the orientation of keypoints links with 3D keypoints.

    # Arguments
        keypoints: Array. 3D keypoints

    # Returns
        orientation: Array. Orientation of keypoint links
    """
    def __init__(self, parents):
        super(ComputeOrientationVector, self).__init__()
        self.parents = parents

    def call(self, keypoints):
        orientation = compute_orientation_vector(keypoints, self.parents)
        return orientation
