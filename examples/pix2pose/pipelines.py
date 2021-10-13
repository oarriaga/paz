import numpy as np
import os
import glob
import random
from tensorflow.keras.utils import Sequence

from paz.abstract import SequentialProcessor, Processor
from paz.abstract.sequence import SequenceExtra
from paz.pipelines import RandomizeRenderedImage
from paz import processors as pr


class GeneratedImageProcessor(Processor):
    """
    Loads pre-generated images
    """
    def __init__(self, path_images, background_images_paths, num_occlusions=1, split=pr.TRAIN, no_ambiguities=False):
        super(GeneratedImageProcessor, self).__init__()
        self.copy = pr.Copy()
        self.augment = RandomizeRenderedImage(background_images_paths, num_occlusions)
        preprocessors_input = [pr.NormalizeImage()]
        preprocessors_output = [NormalizeImageTanh()]
        self.preprocess_input = SequentialProcessor(preprocessors_input)
        self.preprocess_output = SequentialProcessor(preprocessors_output)
        self.split = split

        # Total number of images
        self.num_images = len(glob.glob(os.path.join(path_images, "image_original/*")))

        # Load all images into memory to save time
        self.images_original = [np.load(os.path.join(path_images, "image_original/image_original_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]

        if no_ambiguities:
            self.images_colors = [np.load(os.path.join(path_images, "image_colors_no_ambiguities/image_colors_no_ambiguities_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]
        else:
            self.images_colors = [np.load(os.path.join(path_images, "image_colors/image_colors_{}.npy".format(str(i).zfill(7)))) for i in range(self.num_images)]

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

"""
Creates a batch of train data for the discriminator. For real images the label is 1, 
for fake images the label is 0
"""
def make_batch_discriminator(generator, input_images, color_output_images, label):
    if label == 1:
        return color_output_images, np.ones(len(color_output_images))
    elif label == 0:
        predictions = generator.predict(input_images)
        return predictions[0], np.zeros(len(predictions[0]))


class GeneratingSequencePix2Pose(SequenceExtra):
    """Sequence generator used for generating samples.
    Unfortunately the GeneratingSequence class from paz.abstract cannot be used here. Reason: not all of
    the training data is available right at the start. The error images depend on the predicted color images,
    so that they have to be generated on-the-fly during training. This is done here.

    # Arguments
        processor: Function used for generating and processing ``samples``.
        model: Keras model
        batch_size: Int.
        num_steps: Int. Number of steps for each epoch.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, model, batch_size, num_steps, as_list=False, rotation_matrices=None):
        self.num_steps = num_steps
        self.model = model
        self.rotation_matrices = rotation_matrices
        super(GeneratingSequencePix2Pose, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return self.num_steps

    def rotate_image(self, image, rotation_matrix):
        mask_image = np.ma.masked_not_equal(np.sum(image, axis=-1), -1.*3).mask.astype(float)
        mask_image = np.repeat(mask_image[..., np.newaxis], 3, axis=-1)
        mask_background = np.ones_like(mask_image) - mask_image

        # Rotate the object
        image_rotated = np.einsum('ij,klj->kli', rotation_matrix, image)
        image_rotated *= mask_image
        image_rotated += (mask_background * -1.)

        return image_rotated

    def process_batch(self, inputs, labels, batch_index):
        input_images, samples = list(), list()
        for sample_arg in range(self.batch_size):
            sample = self.pipeline()
            samples.append(sample)
            input_image = sample['inputs'][self.ordered_input_names[0]]
            input_images.append(input_image)

        input_images = np.asarray(input_images)
        # This line is very important. If model.predict(...) is used instead the results are wrong.
        # Reason: BatchNormalization behaves differently, depending on whether it is in train or
        # inference mode. model.predict(...) is the inference mode, so the predictions here will
        # be different from the predictions the model is trained on --> Result: the error images
        # generated here are also wrong
        predictions = self.model(input_images, training=True)

        # Calculate the errors between the target output and the predicted output
        for sample_arg in range(self.batch_size):
            sample = samples[sample_arg]

            # List of tuples of the form (error, error_image)
            stored_errors = []

            # Iterate over all rotation matrices to find the object position
            # with the smallest error
            for rotation_matrix in self.rotation_matrices:
                color_image_rotated = self.rotate_image(sample['labels']['color_output'], rotation_matrix)
                error_image = np.sum(predictions['color_output'][sample_arg] - color_image_rotated, axis=-1, keepdims=True)

                error_value = np.sum(np.abs(error_image))
                stored_errors.append((error_value, error_image))

            # Select the error image with the smallest error
            minimal_error_pair = min(stored_errors, key=lambda t: t[0])
            sample['labels'][self.ordered_label_names[0]] = minimal_error_pair[1]
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)

        return inputs, labels


class NormalizeImageTanh(Processor):
    """
    Normalize image so that the values are between -1 and 1
    """
    def __init__(self):
        super(NormalizeImageTanh, self).__init__()

    def call(self, image):
        return (image/127.5)-1


class DenormalizeImageTanh(Processor):
    """
    Transforms an image from the value range -1 to 1 back to 0 to 255
    """
    def __init__(self):
        super(DenormalizeImageTanh, self).__init__()

    def call(self, image):
        return (image + 1.0)*127.5
