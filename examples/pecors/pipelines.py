import os
import glob
import random
from tqdm import tqdm

import numpy as np
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr

from scenes import SingleView


class GeneratedImageProcessor(Processor):
    """
    Loads pre-generated images
    """
    def __init__(self, path_images, background_images_paths, num_occlusions=1, split=pr.TRAIN, num_stages=6, no_ambiguities=False):
        super(GeneratedImageProcessor, self).__init__()
        self.copy = pr.Copy()
        self.augment = RandomizeRenderedImage(background_images_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.split = split
        self.num_stages = num_stages

        # Total number of images
        self.num_images = len(glob.glob(os.path.join(path_images, "image_original/*")))

        # Load all images into memory to save time
        self.images_original = [np.load(os.path.join(path_images, "image_original/image_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.alpha_original = [np.load(os.path.join(path_images, "alpha_original/alpha_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.images_circle = [np.load(os.path.join(path_images, "image_circle/image_circle_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.images_depth = [np.load(os.path.join(path_images, "image_depth/image_depth_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]


    def call(self):
        index = random.randint(0, self.num_images-1)
        image_original = self.images_original[index]
        image_circle = self.images_circle[index]
        image_depth = self.images_depth[index]
        alpha_original = self.alpha_original[index]

        #image_original = (image_original*255).astype("uint8")

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)

        image_circle = image_circle.astype("float")/255.
        image_depth = image_depth.astype("float") / 255.

        return image_original, image_circle, image_depth


class GeneratedImageGenerator(SequentialProcessor):
    def __init__(self, path_images, background_images_paths, image_size, split=pr.TRAIN, num_occlusions=0):
        super(GeneratedImageGenerator, self).__init__()
        self.add(GeneratedImageProcessor(
            path_images, background_images_paths, num_occlusions, split=split))

        self.add(pr.SequenceWrapper({0: {'input_image': [image_size, image_size, 3]}},
                                    {1: {'circle_output': [image_size, image_size, 3]},
                                     2: {'depth_output': [image_size, image_size, 3]}}))


class GeneratedVectorProcessor(Processor):
    """
    Loads pre-generated images
    """
    def __init__(self, path_images, background_images_paths, num_occlusions=1, split=pr.TRAIN, num_stages=6, no_ambiguities=False):
        super(GeneratedVectorProcessor, self).__init__()
        self.copy = pr.Copy()
        self.augment = RandomizeRenderedImage(background_images_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.split = split
        self.num_stages = num_stages

        # Total number of images
        self.num_images = len(glob.glob(os.path.join(path_images, "image_original/*")))

        # Load all images into memory to save time
        self.images_original = [np.load(os.path.join(path_images, "image_original/image_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.alpha_original = [np.load(os.path.join(path_images, "alpha_original/alpha_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.rotation_vectors = [np.load(os.path.join(path_images, "rotation_vector/rotation_vector_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        self.translation_vectors = [np.load(os.path.join(path_images, "translation_vector/translation_vector_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]


    def call(self):
        index = random.randint(0, self.num_images-1)
        image_original = self.images_original[index]
        rotation_vector = self.rotation_vectors[index]
        translation_vector = self.translation_vectors[index]
        alpha_original = self.alpha_original[index]

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)

        return image_original, rotation_vector, translation_vector


class GeneratedVectorGenerator(SequentialProcessor):
    def __init__(self, path_images, background_images_paths, image_size, split=pr.TRAIN, num_occlusions=0):
        super(GeneratedVectorGenerator, self).__init__()
        self.add(GeneratedVectorProcessor(path_images, background_images_paths, num_occlusions, split=split))
        self.add(pr.SequenceWrapper({0: {'input_image': [image_size, image_size, 3]}},
                                    {1: {'rotation_output': [3, ]},
                                     2: {'translation_output': [3, ]}}))
