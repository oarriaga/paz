from paz.abstract import Processor
import numpy as np


class MakeDictionary(Processor):
    def __init__(self, encoder, renderer):
        super(MakeDictionary, self).__init__()
        self.latent_dimension = encoder.encoder.output_shape[1]
        self.encoder = encoder
        self.renderer = renderer

    def call(self):
        data = self.renderer.render()
        dictionary = {}
        latent_vectors = np.zeros((len(data), self.latent_dimension))
        for sample_arg, sample in enumerate(data):
            image = sample['image']
            latent_vectors[sample_arg] = self.encoder(image)
            dictionary[sample_arg] = image
        dictionary['latent_vectors'] = latent_vectors
        return dictionary


class MeasureSimilarity(Processor):
    def __init__(self, dictionary, measure):
        super(MeasureSimilarity, self).__init__()
        self.dictionary = dictionary
        self.measure = measure

    def call(self, latent_vector):
        latent_vectors = self.dictionary['latent_vectors']
        measurements = self.measure(latent_vectors, latent_vector)
        closest_image = self.dictionary[np.argmax(measurements)]
        return latent_vector, closest_image
