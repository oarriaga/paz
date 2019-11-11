from ..core import Processor
from ..core import ops
import numpy as np


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

    def call(self, kwargs=None):
        image, (matrices, alpha_channel, depth) = self.renderer.render_sample()
        world_to_camera = matrices[0].reshape(4, 4)
        kwargs['image'] = image
        kwargs['world_to_camera'] = world_to_camera
        kwargs['alpha_mask'] = alpha_channel
        kwargs['depth'] = depth
        return kwargs


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

    def call(self, kwargs=None):
        [image_A, image_B], labels = self.renderer.render_sample()
        [matrices, alpha_A, alpha_B] = labels
        # image_A, image_B = image_A / 255.0, image_B / 255.0
        # alpha_A, alpha_B = alpha_A / 255.0, alpha_B / 255.0
        alpha_A = np.expand_dims(alpha_A, -1)
        alpha_B = np.expand_dims(alpha_B, -1)
        alpha_masks = np.concatenate([alpha_A, alpha_B], -1)
        kwargs['matrices'] = matrices
        kwargs['image_A'] = image_A
        kwargs['image_B'] = image_B
        kwargs['alpha_channels'] = alpha_masks
        return kwargs


class ConcatenateAlphaMask(Processor):
    """Concatenate ``alpha_mask`` to ``image``. Useful for changing background.
    """
    def call(self, kwargs):
        image, alpha_mask = kwargs['image'], kwargs['alpha_mask']
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)
        kwargs['image'] = np.concatenate([image, alpha_mask], axis=2)
        return kwargs


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

    def call(self, kwargs):
        world_to_camera = kwargs['world_to_camera']
        keypoints = np.matmul(self.keypoints, world_to_camera.T)
        keypoints = np.expand_dims(keypoints, 0)
        keypoints = self.projector.project(keypoints)[0]
        kwargs['keypoints'] = keypoints
        return kwargs


class DenormalizeKeypoints(Processor):
    """Transform normalized keypoint coordinates into image coordinates
    """
    def __init__(self):
        super(DenormalizeKeypoints, self).__init__()

    def call(self, kwargs):
        keypoints, image = kwargs['keypoints'], kwargs['image']
        height, width = image.shape[0:2]
        keypoints = ops.denormalize_keypoints(keypoints, height, width)
        kwargs['keypoints'] = keypoints
        return kwargs


class NormalizeKeypoints(Processor):
    """Transform keypoints in image coordinates to normalized coordinates
    """
    def __init__(self):
        super(NormalizeKeypoints, self).__init__()

    def call(self, kwargs):
        image, keypoints = kwargs['image'], kwargs['keypoints']
        height, width = image.shape[0:2]
        kwargs['keypoints'] = ops.normalize_keypoints(keypoints, height, width)
        return kwargs


class RemoveKeypointsDepth(Processor):
    """Removes Z component from keypoints.
    """
    def __init__(self):
        super(RemoveKeypointsDepth, self).__init__()

    def call(self, kwargs):
        kwargs['keypoints'] = kwargs['keypoints'][:, :2]
        return kwargs


class PartitionKeypoints(Processor):
    """Partitions keypoints from shape [num_keypoints, 2] into a list of the form
        ((2), (2), ....) and length equal to num_of_keypoints.
        This is performed for tensorflow probablity
    """
    def __init__(self):
        super(PartitionKeypoints, self).__init__()

    def call(self, kwargs):
        keypoints = kwargs['keypoints']
        keypoints = np.vsplit(keypoints, len(keypoints))
        keypoints = [np.squeeze(keypoint) for keypoint in keypoints]
        for keypoint_arg, keypoint in enumerate(keypoints):
            kwargs['keypoint_%s' % keypoint_arg] = keypoint
        return kwargs


class ChangeKeypointsCoordinateSystem(Processor):
    """Changes ``keypoints`` 2D coordinate system using ``box2D`` coordinates
        to locate the new origin at the openCV image origin (top-left).
    """
    def __init__(self):
        super(ChangeKeypointsCoordinateSystem, self).__init__()

    def call(self, kwargs):
        box2D = kwargs['box2D']
        x_min, y_min, x_max, y_max = box2D.coordinates
        keypoints = kwargs['keypoints']
        keypoints[:, 0] = keypoints[:, 0] + x_min
        keypoints[:, 1] = keypoints[:, 1] + y_min
        kwargs['keypoints'] = keypoints
        return kwargs
