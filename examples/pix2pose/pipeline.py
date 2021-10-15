from paz.abstract import SequentialProcessor
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz import processors as pr
from .processors import ImageToClosedOneBall


class AutoEncoderDomainRandomization(SequentialProcessor):
    """Performs domain randomization on a rendered image
    """
    def __init__(self, image_shape, image_paths, num_occlusions=1):
        super(AutoEncoderDomainRandomization, self).__init__()
        self.add(pr.Render())
        self.add(pr.ControlMap(pr.Copy(), [0], [2], keep={0: 0}))
        self.add(pr.ControlMap(RandomizeRender(image_paths), [0, 1], [0]))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        self.add(pr.ControlMap(ImageToClosedOneBall(), [1], [1]))
        self.add(pr.SequenceWrapper({0: {'input_image': image_shape}},
                                    {1: {'label_image': image_shape}}))
