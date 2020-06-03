import numpy as np

from ..abstract import Processor
from ..backend.keypoints import normalize_keypoints
from ..backend.keypoints import denormalize_keypoints


class Render(Processor):
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def call(self):
        return self.renderer.render()


class RenderSingleViewSample(Processor):
    """Renders a batch of images and puts them in the selected topic
    # Arguments
        renderer: Python object with ``render_sample'' method. This method
            should render images and some labels of the image e.g.
                matrices, depth, alpha_channel
            It should output a list of length two containing a numpy
            array of the image and a list having the labels in the
            following order
                (matrices, alpha_channel, depth_image)
            Renderers are available in poseur.
    """
    def __init__(self, renderer):
        self.renderer = renderer
        super(RenderSingleViewSample, self).__init__()

    def call(self):
        image, (matrices, alpha_channel, depth) = self.renderer.render_sample()
        world_to_camera = matrices[0].reshape(4, 4)
        return image, alpha_channel, depth, world_to_camera


class RenderMultiViewSample(Processor):
    """Renders a batch of images and puts them in the selected topic
    # Arguments
        renderer: Python object with ``render_sample'' method. This method
            should render images and some labels of the image e.g.
                matrices, depth, alpha_channel
            It should output a list of length two containing a numpy
            array of the image and a list having the labels in the
            following order
                (matrices, alpha_channel, depth_image)
            Renderers are available in poseur.
    """
    def __init__(self, renderer):
        self.renderer = renderer
        super(RenderMultiViewSample, self).__init__()

    def call(self):
        [image_A, image_B], labels = self.renderer.render_sample()
        [matrices, alpha_A, alpha_B] = labels
        alpha_A = np.expand_dims(alpha_A, -1)
        alpha_B = np.expand_dims(alpha_B, -1)
        alpha_masks = np.concatenate([alpha_A, alpha_B], -1)
        return image_A, image_B, alpha_masks, matrices


class ConcatenateAlphaMask(Processor):
    """Concatenate ``alpha_mask`` to ``image``. Useful for changing background.
    """
    def call(self, image, alpha_mask):
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)
        image = np.concatenate([image, alpha_mask], axis=2)
        return image


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
    def __init__(self, image_size):
        self.image_size = image_size
        super(DenormalizeKeypoints, self).__init__()

    def call(self, keypoints, image):
        height, width = self.image_size[0:2]
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
