from functools import partial
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from paz.datasets.omniglot import (
    load, split_data, sample_between_alphabet)

from cnn import CONVNET, MAML, Predict


seed = 777
RNG = np.random.default_rng(seed)
num_steps = 60_000
batch_size = 10
min_amplitude = 0.1
max_amplitude = 5.0
min_x = -5.0
max_x = 5.0
train_episodes = 1_000
test_episodes = 10
weights_filename = f'MAML_CNN_epochs-{num_steps}.hdf5'

image_shape = (28, 28, 1)
validation_split = 0.20


compute_loss = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=0.003)
learning_rate = 0.5

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
losses = fit(RNG, train_sampler, num_steps)
meta_model.save_weights(weights_filename)

tests_data = load('test', image_shape[:2], True)
tests_sampler = partial(sample_between_alphabet, RNG, tests_data, *train_args)
((x_true_support, y_true_support),
 (x_true_queries, y_true_queries)) = tests_sampler()

predict = Predict(meta_model, learning_rate, compute_loss)
y_pred_queries = predict(x_true_support, y_true_support, x_true_queries, 100)
y_pred_queries = np.argmax(y_pred_queries, axis=1)


def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
