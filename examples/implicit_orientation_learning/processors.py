from paz.abstract import Processor
from paz.backend.image import blend_alpha_channel, random_image_crop
from paz.backend.image import make_random_plain_image, concatenate_alpha_mask
from paz.backend.image import draw_filled_polygon, load_image
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


class AlphaBlending(Processor):
    def __init__(self):
        super(AlphaBlending, self).__init__()

    def call(self, image, background):
        return blend_alpha_channel(image, background)


class RandomImageCrop(Processor):
    def __init__(self, size):
        super(RandomImageCrop, self).__init__()
        self.size = size

    def call(self, image):
        return random_image_crop(image, self.size)


class MakeRandomPlainImage(Processor):
    def __init__(self, shape):
        super(MakeRandomPlainImage, self).__init__()
        self.shape = shape

    def call(self):
        return make_random_plain_image(self.shape)


class ConcatenateAlphaMask(Processor):
    def __init__(self, **kwargs):
        super(ConcatenateAlphaMask, self).__init__(**kwargs)

    def call(self, image, alpha_mask):
        return concatenate_alpha_mask(image, alpha_mask)


class BlendRandomCroppedBackground(Processor):
    def __init__(self, background_paths):
        super(BlendRandomCroppedBackground, self).__init__()
        if not isinstance(background_paths, list):
            raise ValueError('``background_paths`` must be list')
        if len(background_paths) == 0:
            raise ValueError('No paths given in ``background_paths``')
        self.background_paths = background_paths

    def call(self, image):
        random_arg = np.random.randint(0, len(self.background_paths))
        background_path = self.background_paths[random_arg]
        background = load_image(background_path)
        background = random_image_crop(background, image.shape[:2])
        if background is None:
            background = make_random_plain_image(image.shape[:2])
        return blend_alpha_channel(image, background)


class AddOcclusion(Processor):
    def __init__(self, max_radius_scale=0.5, probability=0.5):
        """TODO: add use of probability
        """
        super(AddOcclusion, self).__init__()
        self.max_radius_scale = max_radius_scale

    def _random_vertices(self, center, max_radius, min_vertices, max_vertices):
        num_vertices = np.random.randint(min_vertices, max_vertices)
        angle_delta = 2 * np.pi / num_vertices
        initial_angle = np.random.uniform(0, 2 * np.pi)
        angles = initial_angle + np.arange(0, num_vertices) * angle_delta
        x_component = np.cos(angles).reshape(-1, 1)
        y_component = np.sin(angles).reshape(-1, 1)
        vertices = np.concatenate([x_component, y_component], -1)
        random_lengths = np.random.uniform(0, max_radius, num_vertices)
        random_lengths = random_lengths.reshape(num_vertices, 1)
        vertices = vertices * random_lengths
        vertices = vertices + center
        return vertices.astype(np.int32)

    def add_occlusion(self, image, max_radius_scale):
        height, width = image.shape[:2]
        max_radius = np.max((height, width)) * max_radius_scale
        center = np.random.rand(2) * np.array([width, height])
        vertices = self._random_vertices(center, max_radius, 3, 7)
        color = np.random.randint(0, 256, 3).tolist()
        return draw_filled_polygon(image, vertices, color)

    def call(self, image):
        return self.add_occlusion(image, self.max_radius_scale)
