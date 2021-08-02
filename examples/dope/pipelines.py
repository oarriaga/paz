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


class ImageGeneratorProcessor(Processor):
    def __init__(self, renderer, image_paths, num_occlusions=1, split=pr.TRAIN, num_stages=3):
        super(ImageGeneratorProcessor, self).__init__()
        self.copy = pr.Copy()
        self.render = pr.Render(renderer)

        if not (image_paths is None):
            self.augment = RandomizeRenderedImage(image_paths, num_occlusions)
        else:
            self.augment = None

        preprocessors_input = [pr.NormalizeImage()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.split = split
        self.num_stages = num_stages

    def call(self):
        image_original, alpha_original, bounding_box_points, belief_maps, create_affinity_maps, _ = self.render()

        if not (self.augment is None):
            if self.split == pr.TRAIN:
                image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)

        # Just works for one object (for now)
        belief_maps = np.transpose(belief_maps[0], axes=(1,2,0))
        #affinity_maps = np.transpose(create_affinity_maps[0], axes=(1,2,0))
        affinity_maps = list()

        if self.num_stages == 1:
            return image_original, belief_maps, affinity_maps

        if self.num_stages == 2:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps

        if self.num_stages == 3:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps

        if self.num_stages == 4:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps

        if self.num_stages == 5:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps

        return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps


class ImageGenerator(SequentialProcessor):
    def __init__(self, renderer, size_normal, size_downscaled, image_paths, num_occlusions=1, split=pr.TRAIN, num_stages=3):
        super(ImageGenerator, self).__init__()
        self.add(ImageGeneratorProcessor(renderer, image_paths, num_occlusions, split, num_stages))

        outputs = dict()
        for i in range(2, 2*num_stages+1, 2):
            outputs[i-1] = {'belief_maps_stage_' + str(int(i/2)): [size_downscaled, size_downscaled, 9]}
            #outputs[i] = {'affinity_maps_stage_' + str(int(i/2)): [size_downscaled, size_downscaled, 16]}

        self.add(pr.SequenceWrapper({0: {'input_1': [size_normal, size_normal, 3]}}, outputs))


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

        if no_ambiguities:
            self.belief_maps = [np.load(os.path.join(path_images, "belief_maps_no_ambiguities/belief_maps_no_ambiguities_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        else:
            self.belief_maps = [np.load(os.path.join(path_images, "belief_maps/belief_maps_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]

    def call(self):
        index = random.randint(0, self.num_images-1)
        image_original = self.images_original[index]
        belief_maps = self.belief_maps[index]
        alpha_original = self.alpha_original[index]

        if self.split == pr.TRAIN:
            image_original = self.augment(image_original, alpha_original)

        image_original = self.preprocess_input(image_original)

        # Just works for one object (for now)
        belief_maps = np.transpose(belief_maps[0], axes=(1, 2, 0))
        #affinity_maps = np.transpose(create_affinity_maps[0], axes=(1,2,0))
        affinity_maps = list()

        if self.num_stages == 1:
            return image_original, belief_maps, affinity_maps

        if self.num_stages == 2:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps

        if self.num_stages == 3:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps

        if self.num_stages == 4:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps

        if self.num_stages == 5:
            return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps

        return image_original, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps, belief_maps, affinity_maps


class GeneratedImageGenerator(SequentialProcessor):
    def __init__(self, path_images, background_images_paths, num_occlusions, image_size_input, image_size_output, num_stages, split=pr.TRAIN):
        super(GeneratedImageGenerator, self).__init__()
        self.add(GeneratedImageProcessor(
            path_images, background_images_paths, num_occlusions, num_stages=num_stages, split=split))

        outputs = dict()
        for i in range(2, 2*num_stages+1, 2):
            outputs[i-1] = {'belief_maps_stage_' + str(int(i/2)): [image_size_output, image_size_output, 9]}

        self.add(pr.SequenceWrapper({0: {'input_1': [image_size_input, image_size_input, 3]}}, outputs))


def generate_train_data(num_images, save_path, obj_path, images_directory, image_size, scaling_factor):
    """
    Generate training images, so that they do not have to be generated at train time
    """
    batch_save_size = 1000

    colors = [np.array([255, 0, 0]), np.array([0, 255, 0])]
    renderer = SingleView(filepath=obj_path, colors=colors, viewport_size=(image_size, image_size),
                          y_fov=3.14159 / 4.0, distance=[0.3, 0.5], light_bounds=[.5, 30], top_only=bool(0),
                          roll=3.14159, shift=0.05)

    processor = ImageGeneratorProcessor(renderer, None, num_occlusions=0)

    images_list, belief_maps_list, affinity_maps_list = list(), list(), list()

    for i in tqdm(range(num_images)):
        image_original, belief_maps, affinity_maps, _, _, _, _ = processor.call()

        # Bring image to range 0 to 255 to save it as uint8
        image_original = (image_original*255).astype(np.uint8)

        images_list.append(image_original)
        belief_maps_list.append(belief_maps)
        #affinity_maps_list.append(affinity_maps)

        if i%batch_save_size == 0 and i > 0:
            np.save(os.path.join(save_path, "images_batch_{}.npy".format(int(i/batch_save_size))), np.array(images_list))
            np.save(os.path.join(save_path, "belief_maps_batch_{}.npy".format(int(i/batch_save_size))), np.array(belief_maps_list))
            #np.save(os.path.join(save_path, "affinity_maps_{}.npy".format(int(i/batch_save_size))), np.array(affinity_maps_list))

            images_list, belief_maps_list, affinity_maps_list = list(), list(), list()


if __name__ == "__main__":
    generate_train_data(30000, "/media/fabian/Data/Masterarbeit/data/dope/train_data_224px", ["/home/fabian/.keras/datasets/035_power_drill/tsdf/textured.obj"], "", 224, 8.0)