from paz.core import Processor, SequentialProcessor
from paz import processors as pr
import numpy as np


class MeasureSimilarity(Processor):
    def __init__(self, dictionary, measure, input_topic='latent_vector',
                 label_topic='dictionary_image'):
        super(MeasureSimilarity, self).__init__()
        self.dictionary = dictionary
        self.measure = measure
        self.input_topic = input_topic
        self.label_topic = label_topic

    def call(self, kwargs):
        latent_vector = kwargs[self.input_topic]
        latent_vectors = self.dictionary['latent_vectors']
        measurements = self.measure(latent_vectors, latent_vector)
        best_arg = np.argmax(measurements)
        best_image = self.dictionary[best_arg]
        kwargs[self.label_topic] = best_image
        return kwargs


class Normalize(Processor):
    def __init__(self, topic):
        super(Normalize, self).__init__()
        self.topic = topic

    def call(self, kwargs):
        data = kwargs[self.topic]
        kwargs[self.topic] = data / np.linalg.norm(data)
        return kwargs


class AlphaBlending(Processor):
    def __init__(self):
        super(AlphaBlending, self).__init__()

    def call(self, image, background):
        if image.shape[-1] != 4:
            raise ValueError('``image`` does not contain an alpha mask.')
        foreground, alpha = np.split(image, [3], -1)
        return (1.0 - (alpha / 255.0)) * background.astype(float)


class RandomImageCrop(Processor):
    def __init__(self, size):
        super(RandomImageCrop, self).__init__()
        self.size = size

    def call(self, image):
        H, W = image.shape[:2]
        if (self.box_size >= H) or (self.box_size >= W):
            print('WARNING: Image is smaller than crop size')
            return None
        x_min = np.random.randint(0, (W - 1) - self.size)
        y_min = np.random.randint(0, (H - 1) - self.size)
        x_max = int(x_min + self.size)
        y_max = int(y_min + self.size)
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image


class MakeRandomPlainImage(Processor):
    def __init__(self, shape):
        super(MakeRandomPlainImage, self).__init__()
        self.H, self.W, self.num_channels = shape

    def call(self):
        random_RGB = np.random.randint(0, 256, self.num_channels)
        return np.ones((self.H, self.W, self.num_channels)) * random_RGB


class AddCroppedBackground(Processor):
    def __init__(self, image_paths, size):
        super(AddCroppedBackground, self).__init__()
        if not isinstance(image_paths, list):
            raise ValueError('``image_paths`` must be list')
        if len(image_paths) == 0:
            raise ValueError('No paths given in ``image_paths``')

        self.image_paths = image_paths
        self.build_background = SequentialProcessor(pr.LoadImage())
        self.build_background.add(pr.LoadImage())
        self.build_background.add(RandomImageCrop(size))
        self.alpha_blend = AlphaBlending()
        self.make_random_plain_image = MakeRandomPlainImage()

    def call(self, image):
        random_arg = np.random.randint(0, len(self.image_paths))
        image_path = self.image_paths[random_arg]
        background = self.build_background(image_path)
        if background is None:
            background = self.make_random_plain_image()
        return self.alpha_blend(image, background)
