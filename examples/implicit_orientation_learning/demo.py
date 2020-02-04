import os
import json
import numpy as np
from tensorflow.keras.utils import get_file

from sklearn.metrics.pairwise import cosine_similarity as measure

from paz.core import VideoPlayer
from paz.core import SequentialProcessor
from paz import processors as pr

from poseur.scenes import DictionaryViews

from model import AutoEncoder
from pipelines import ImplicitRotationInference
from processors import Normalize

# model_name = 'VanillaAutoencoder128_128_Switch_1_scaled'
model_name = 'VanillaAutoencoder128_128_035_power_drill'
path = os.path.join(os.path.expanduser('~'), '.keras/paz/models/', model_name)
parameters = json.load(open(os.path.join(path, 'hyperparameters.json'), 'r'))
size = parameters['image_size']
latent_dimension = parameters['latent_dimension']
input_shape = (size, size, 3)
weights_path = os.path.join(path, model_name + '_weights.hdf5')
OBJ_file = get_file('textured.obj', None,
                    cache_subdir='paz/datasets/ycb/models/035_power_drill/')


viewport_size, y_fov = (128, 128), 3.1416 / 4.0
distance, shift, light, background = 0.3, 0.01, 10, 0
roll, translate, sphere = 3.14159, 0.01, 'full'
scene = DictionaryViews(OBJ_file, viewport_size, y_fov, distance,
                        sphere, roll, light, background, True, 10, 10)
data = scene.render_dictionary()


class EncoderInference(SequentialProcessor):
    def __init__(self, encoder, input_topic, label_topic):
        super(EncoderInference, self).__init__()
        pipeline = [pr.ConvertColor('RGB', to='BGR'),
                    pr.ResizeImage(encoder.input_shape[1:3]),
                    pr.NormalizeImage(input_topic),
                    pr.ExpandDims(0, input_topic)]
        self.add(pr.Predict(encoder, input_topic, label_topic, pipeline))
        self.add(pr.Squeeze(0, label_topic))
        # self.add(Normalize(label_topic))


# TODO you can add this as a method of the original pipeline
encoder = AutoEncoder(input_shape, latent_dimension, mode='encoder')
encoder.load_weights(weights_path, by_name=True)

decoder = AutoEncoder(input_shape, latent_dimension, mode='decoder')
decoder.load_weights(weights_path, by_name=True)

pipeline = EncoderInference(encoder, 'image', 'latent_vector')


def make_dictionary(pipeline, data):
    dictionary = {}
    latent_vectors = np.zeros((len(data), latent_dimension))
    for sample_arg, sample in enumerate(data):
        processed_sample = pipeline(sample)
        latent_vector = processed_sample['latent_vector']
        latent_vectors[sample_arg] = latent_vector
        dictionary[sample_arg] = processed_sample['image']
    dictionary['latent_vectors'] = latent_vectors
    return dictionary


dictionary = make_dictionary(pipeline, data)
pipeline = ImplicitRotationInference(encoder, decoder, measure, dictionary)
player = VideoPlayer((1280, 960), pipeline, camera=2)
player.start()
