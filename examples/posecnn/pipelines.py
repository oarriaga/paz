import numpy as np
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr


class ImageGeneratorProcessor(Processor):
    def __init__(self, renderer, image_paths, num_occlusions=1, split=pr.TRAIN):
        super(ImageGeneratorProcessor, self).__init__()
        self.copy = pr.Copy()
        self.render = pr.Render(renderer)
        self.augment = RandomizeRenderedImage(image_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.split = split

    def call(self):
        image_original, alpha_original, semantic_segmentation_image, distance_x_direction, distance_y_direction, depth_image = self.render()

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)

        return image_original, semantic_segmentation_image, distance_x_direction, distance_y_direction, depth_image


class ImageGenerator(SequentialProcessor):
    def __init__(self, renderer, size, image_paths, num_occlusions=1, split=pr.TRAIN):
        super(ImageGenerator, self).__init__()
        self.add(ImageGeneratorProcessor(renderer, image_paths, num_occlusions, split))
        self.add(pr.SequenceWrapper(
            {0: {'input_1': [size, size, 3]}},
            {1: {'log_softmax_out': [size, size, 3]}}))