import numpy as np

from ..abstract import Processor
from ..backend.keypoints import normalize_keypoints
from ..backend.keypoints import denormalize_keypoints


class ProjectKeypoints(Processor):
    """Renders a batch of images and puts them in the selected topic
    # Arguments
        projector:
        keypoints:
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


class DenormalizeKeypoints(Processor):
    """Transform normalized keypoint coordinates into image coordinates
    # Arguments
        image_size: List of two floats having height and width of image.
    """
    def __init__(self):
        super(DenormalizeKeypoints, self).__init__()

    def call(self, keypoints, image):
        height, width = image.shape[0:2]
        keypoints = denormalize_keypoints(keypoints, height, width)
        return keypoints


class NormalizeKeypoints(Processor):
    """Transform keypoints in image coordinates to normalized coordinates
    """
    def __init__(self, image_size):
        self.image_size = image_size
        super(NormalizeKeypoints, self).__init__()

    def call(self, keypoints):
        height, width = self.image_size[0:2]
        keypoints = normalize_keypoints(keypoints, height, width)
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
        This is performed for tensorflow probablity
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
