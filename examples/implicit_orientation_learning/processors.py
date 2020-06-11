from paz.abstract import Processor, SequentialProcessor
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


class ConcatenateAlphaMask(Processor):
    def __init__(self, **kwargs):
        super(ConcatenateAlphaMask, self).__init__(**kwargs)

    def call(self, image, alpha_mask):
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)
        return np.concatenate([image, alpha_mask], axis=2)


class MakeRandomPlainImage(Processor):
    def __init__(self, shape):
        super(MakeRandomPlainImage, self).__init__()
        self.H, self.W, self.num_channels = shape

    def call(self):
        random_RGB = np.random.randint(0, 256, self.num_channels)
        return np.ones((self.H, self.W, self.num_channels)) * random_RGB


class AddOcclusion(Processor):
    def __init__(self, max_radius_scale=.5, probability=.5):
        super(AddOcclusion, self).__init__()
        self.max_radius_scale = max_radius_scale

    def call(self, image):
        height, width = image.shape[:2]
        max_distance = np.max((height, width)) * self.max_radius_scale
        num_vertices = np.random.randint(3, 7)
        angle_between_vertices = 2 * np.pi / num_vertices
        initial_angle = np.random.uniform(0, 2 * np.pi)
        center = np.random.rand(2) * np.array([width, height])
        vertices = np.zeros((num_vertices, 2), dtype=np.int32)
        for vertex_arg in range(num_vertices):
            angle = initial_angle + (vertex_arg * angle_between_vertices)
            vertex = np.array([np.cos(angle), np.sin(angle)])
            vertex = np.random.uniform(0, max_distance) * vertex
            vertices[vertex_arg] = (vertex + center).astype(np.int32)
        color = np.random.randint(0, 256, 3).tolist()
        draw_filled_polygon(image, vertices, color)
        return image



class AddCroppedBackground(Processor):
    def __init__(self, image_paths, size):
        super(AddCroppedBackground, self).__init__()
        if not isinstance(image_paths, list):
            raise ValueError('``image_paths`` must be list')
        if len(image_paths) == 0:
            raise ValueError('No paths given in ``image_paths``')

        self.image_paths = image_paths
        self.build_background = SequentialProcessor()
        self.build_background.add(pr.LoadImage())
        self.build_background.add(RandomImageCrop(size))
        self.alpha_blend = AlphaBlending()
        self.make_random_plain_image = MakeRandomPlainImage((size, size, 3))

    def call(self, image):
        random_arg = np.random.randint(0, len(self.image_paths))
        image_path = self.image_paths[random_arg]
        background = self.build_background(image_path)
        if background is None:
            background = self.make_random_plain_image()
        return self.alpha_blend(image, background)
