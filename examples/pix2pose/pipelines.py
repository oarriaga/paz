import numpy as np
import os
import glob
import random
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

from paz.abstract import SequentialProcessor, Processor
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr


class DepthImageGeneratorProcessor(Processor):
    def __init__(self, renderer, image_paths, num_occlusions=1, split=pr.TRAIN):
        super(DepthImageGeneratorProcessor, self).__init__()
        self.copy = pr.Copy()
        self.render = pr.Render(renderer)
        self.augment = RandomizeRenderedImage(image_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        preprocessors_output = [pr.NormalizeImageTanh()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.preprocess_output = SequentialProcessor(preprocessors_output)
        self.split = split

    def call(self):
        image_original, image_colors, alpha_original = self.render()

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)
        image_colors = self.preprocess_output(image_colors)

        return image_original, image_colors


class GeneratedImageProcessor(Processor):
    """
    Loads pre-generated images
    """
    def __init__(self, path_images, background_images_paths, num_occlusions=1, split=pr.TRAIN, no_ambiguities=False):
        super(GeneratedImageProcessor, self).__init__()
        self.copy = pr.Copy()
        self.augment = RandomizeRenderedImage(background_images_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        preprocessors_output = [pr.NormalizeImageTanh()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.preprocess_output = SequentialProcessor(preprocessors_output)
        self.split = split

        # Total number of images
        self.num_images = len(glob.glob(os.path.join(path_images, "image_original/*")))

        # Load all images into memory to save time
        self.images_original = [np.load(os.path.join(path_images, "image_original/images_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]

        if no_ambiguities:
            self.images_colors = [np.load(os.path.join(path_images, "image_colors_no_ambiguities/image_colors_no_ambiguities_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        else:
            self.images_colors = [np.load(os.path.join(path_images, "image_colors/images_colors_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]

        self.alpha_original = [np.load(os.path.join(path_images, "alpha_original/alpha_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]


    def call(self):
        index = random.randint(0, self.num_images-1)
        image_original = self.images_original[index]
        image_colors = self.images_colors[index]
        alpha_original = self.alpha_original[index]

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)
        image_colors = self.preprocess_output(image_colors)

        return image_original, image_colors


class GeneratedImageGenerator(SequentialProcessor):
    def __init__(self, path_images, size, background_images_paths, num_occlusions=1, split=pr.TRAIN):
        super(GeneratedImageGenerator, self).__init__()
        self.add(GeneratedImageProcessor(
            path_images, background_images_paths, num_occlusions, split))
        self.add(pr.SequenceWrapper(
            {0: {'input_image': [size, size, 3]}},
            {1: {'color_output': [size, size, 3]}, 0: {'error_output': [size, size, 1]}}))


class DepthImageGenerator(SequentialProcessor):
    def __init__(self, renderer, size, image_paths, num_occlusions=1, split=pr.TRAIN):
        super(DepthImageGenerator, self).__init__()
        self.add(DepthImageGeneratorProcessor(
            renderer, image_paths, num_occlusions, split))
        self.add(pr.SequenceWrapper(
            {0: {'input_image': [size, size, 3]}},
            {1: {'color_output': [size, size, 3]}, 0: {'error_output': [size, size, 1]}}))


def make_batch_discriminator(generator, input_images, color_output_images, label):
    if label == 1:
        return color_output_images, np.ones(len(color_output_images))
    elif label == 0:
        predictions = generator.predict(input_images)
        return predictions[0], np.zeros(len(predictions[0]))


class RendererDataGenerator(Sequence):

    def __init__(self, renderer, steps_per_epoch, batch_size=32):
        self.renderer = renderer
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        list_rgb_images, list_depth_images = list(), list()
        for _ in range(self.batch_size):
            image, alpha, depth = self.renderer.render()
            list_rgb_images.append(image)
            list_depth_images.append(depth)

        X = np.asarray(list_rgb_images)
        y = np.asarray(list_depth_images)
        return X, y
