from paz.abstract import SequentialProcessor
from paz.pipelines import RandomizeRenderedImage as RandomizeRender
from paz import processors as pr
# from processors import ImageToClosedOneBall


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
        self.add(pr.SequenceWrapper({0: {'input_image': [H, W, 3]}},
                                    {1: {'label_image': [H, W, 4]}}))
