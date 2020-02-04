from paz.core import Processor
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
