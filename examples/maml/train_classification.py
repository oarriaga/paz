from functools import partial
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from paz.datasets.omniglot import (
    load, split_data, sample_between_alphabet)

from cnn import CONVNET, MAML


seed = 777
compute_loss = SparseCategoricalCrossentropy()
optimizer = Adam()
learning_rate = 0.01
RNG = np.random.default_rng(seed)
epochs = 20
batch_size = 10
min_amplitude = 0.1
max_amplitude = 5.0
min_x = -5.0
max_x = 5.0
train_episodes = 1_000
test_episodes = 10
weights_filename = f'MAML_CNN_epochs-{epochs}.hdf5'

image_shape = (28, 28, 1)
validation_split = 0.20

train_ways = 5
train_shots = 10
train_queries = 10
train_args = (train_ways, train_shots, train_queries)
meta_model = CONVNET(train_ways, image_shape, 4)


train_data = load('train', image_shape[:2], True)
train_data, validation_data = split_data(train_data, validation_split)

train_sampler = partial(sample_between_alphabet, RNG, train_data, *train_args)
(x1, y1), (x2, y2) = train_sampler()

fit = MAML(meta_model, compute_loss, optimizer, learning_rate)
# train_samplers = [train_sampler for _ in range(100)]
losses = fit(RNG, train_sampler, epochs)
# meta_model.save_weights(weights_filename)
