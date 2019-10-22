from core import Processor
import numpy as np


class RenderSample(Processor):
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
        batch_size: Integer. Number of images to be rendered
    """
    def __init__(self, renderer, batch_size):
        self.renderer = renderer
        self.batch_size = batch_size
        super(RenderSample, self).__init__()

    def call(self, kwargs):
        image, (matrices, alpha_channel, depth) = self.renderer.render_sample()
        world_to_camera = matrices[0].reshape(4, 4)
        kwargs['image'] = image
        kwargs['world_to_camera'] = world_to_camera
        kwargs['alpha_mask'] = alpha_channel
        kwargs['depth'] = depth
        return kwargs


class ConcatenateAlphaMask(Processor):
    """Concatenate ``alpha_mask`` to ``image``. Useful for changing background.
    """
    def call(self, kwargs):
        image, alpha_mask = kwargs['image'], kwargs['alpha_mask']
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
