import glob
import os
import numpy as np
import random

from paz.abstract import Processor, SequentialProcessor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr

class GeneratedImageProcessor(Processor):
    """
    Loads pre-generated images
    """
    def __init__(self, path_images, background_images_paths, num_occlusions=1, split=pr.TRAIN):
        super(GeneratedImageProcessor, self).__init__()
        self.copy = pr.Copy()
        self.augment = RandomizeRenderedImage(background_images_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.split = split

        # Total number of images
        self.num_images = len(glob.glob(os.path.join(path_images, "image_original/*")))

        # Load all images into memory to save time
        self.images_original = [np.load(os.path.join(path_images, "image_original/image_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.alpha_original = [np.load(os.path.join(path_images, "alpha_original/alpha_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.epos_output = [np.load(os.path.join(path_images, "epos_output/epos_output_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]

    def call(self):
        index = random.randint(0, self.num_images-1)
        image_original = self.images_original[index]
        epos_output = self.epos_output[index]
        alpha_original = self.alpha_original[index]

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)
        #epos_output = self.preprocess_input(epos_output)

        return image_original, epos_output


class GeneratedImageGenerator(SequentialProcessor):
    def __init__(self, path_images, background_images_paths, image_size, output_channels, split=pr.TRAIN, num_occlusions=0):
        super(GeneratedImageGenerator, self).__init__()
        self.add(GeneratedImageProcessor(path_images, background_images_paths, num_occlusions, split=split))
        self.add(pr.SequenceWrapper({0: {'input_image': [image_size, image_size, 3]}}, {1: {'output': [image_size, image_size, output_channels]}}))
