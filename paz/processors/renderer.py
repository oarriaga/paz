import numpy as np
from ..abstract import Processor


class Render(Processor):
    """Render images and labels.

    # Arguments
        renderer: Object that renders images and labels using a method
            ''render_sample()''.
    """
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def call(self):
        return self.renderer.render_sample()


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
