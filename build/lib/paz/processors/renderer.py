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
        return self.renderer.render()
