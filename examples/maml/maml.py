import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input, Dense, InputLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import MeanSquaredError


def MLP(hidden_size=40):
    inputs = Input((1,), name='inputs')
    x = Dense(hidden_size, activation='relu')(inputs)
    x = Dense(hidden_size, activation='relu')(x)
    outputs = Dense(1, name='outputs')(x)
    return Model(inputs=inputs, outputs=outputs, name='MLP')


def shuffle(RNG, dataset):
    RNG.shuffle(dataset)
    return dataset


def layer_arg_to_gradient_arg(layer_arg):
    if layer_arg == 0:
        raise ValueError()
    kernel_arg = 2 * (layer_arg - 1)
    biases_arg = kernel_arg + 1
    return kernel_arg, biases_arg


def copy_model(model):
    meta_weights = model.get_weights()
    copied_model = clone_model(model)
    copied_model.set_weights(meta_weights)
    return copied_model


def gradient_step(learning_rate, gradients, parameters_old):
    return tf.subtract(parameters_old, tf.multiply(learning_rate, gradients))


def meta_to_task(meta_model, support_gradients, learning_rate):
    task_model = copy_model(meta_model)
    for layer_arg in range(len(meta_model.layers)):
        if isinstance(meta_model.layers[layer_arg], InputLayer):
            continue
        kernel_arg, biases_arg = layer_arg_to_gradient_arg(layer_arg)
        kernel_gradients = support_gradients[kernel_arg]
        biases_gradients = support_gradients[biases_arg]

        layer = meta_model.layers[layer_arg]
        kernel_args = (learning_rate, kernel_gradients, layer.kernel)
        biases_args = (learning_rate, biases_gradients, layer.bias)
        task_model.layers[layer_arg].kernel = gradient_step(*kernel_args)
        task_model.layers[layer_arg].bias = gradient_step(*biases_args)
    return task_model


def to_tensor(x, y):
    x_tensor = tf.convert_to_tensor(x)
    y_tensor = tf.convert_to_tensor(y)
    return x_tensor, y_tensor


def MAML(meta_model, compute_loss, optimizer, learning_rate=0.01):
    def fit(RNG, dataset, epochs):
        losses = []
        for epoch_arg in range(epochs):
            epoch_loss = 0
            for step, sampler in enumerate(shuffle(RNG, dataset)):
                x_true_support, y_true_support = to_tensor(*sampler())
                x_true_queries, y_true_queries = to_tensor(*sampler())
                with tf.GradientTape() as meta_tape:
                    with tf.GradientTape() as task_tape:
                        y_pred = meta_model(x_true_support, training=True)
                        support_loss = compute_loss(y_true_support, y_pred)
                    support_gradients = task_tape.gradient(
                        support_loss, meta_model.trainable_variables)
                    task_model = meta_to_task(
                        meta_model, support_gradients, learning_rate)
                    y_task_pred = task_model(x_true_queries, training=True)
                    task_loss = compute_loss(y_true_queries, y_task_pred)
                meta_weights = meta_model.trainable_variables
                gradients = meta_tape.gradient(task_loss, meta_weights)
                optimizer.apply_gradients(zip(gradients, meta_weights))
                epoch_loss = epoch_loss + task_loss
            epoch_loss = epoch_loss / len(dataset)
            print('epoch {} | loss = {}'.format(epoch_arg, epoch_loss))
        return losses
    return fit


def Predict(model, learning_rate, compute_loss):
    def call(x_support, y_support, x_queries, num_steps):
        model_copy = copy_model(model)
        model_copy.compile(SGD(learning_rate), compute_loss)
        for step in range(num_steps):
            model_copy.fit(x_support, y_support)
        y_queries_pred = model_copy(x_queries)
        return y_queries_pred
    return call


def build_equally_spaced_points(num_points):
    return np.linspace(min_x, max_x, num_points)


def sample_random_points(RNG, num_points, min_x, max_x):
    return RNG.uniform(min_x, max_x, num_points)


def sample_amplitude(RNG, min_amplitude=0.1, max_amplitude=5.0):
    return RNG.uniform(min_amplitude, max_amplitude)


def sample_phase(RNG):
    return RNG.uniform(0, np.pi)


def compute_sinusoid(x, amplitude, phase):
    return amplitude * np.sin(x - phase)


def Sinusoid(RNG, num_points, min_amplitude=0.1, max_amplitude=5.0):
    amplitude = sample_amplitude(RNG, min_amplitude, max_amplitude)
    phase = sample_phase(RNG)

    def sample(batch_size=None, equally_spaced=False):
        batch_size = num_points if batch_size is None else batch_size
        if equally_spaced:
            x = build_equally_spaced_points(batch_size)
        else:
            x = sample_random_points(RNG, batch_size, min_x, max_x)
        y = compute_sinusoid(x, amplitude, phase)
        return x, y
    return sample


class Generator(Sequence):
    def __init__(self, samplers):
        self.samplers = samplers

    def __len__(self):
        return len(self.samplers)

    def __getitem__(self, idx):
        x, y = self.samplers[idx]()
        return {'inputs': x}, {'outputs': y}



seed = 777
meta_model = MLP()
compute_loss = MeanSquaredError()
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
weights_filename = f'MAML_MLP_epochs-{epochs}.hdf5'


train_samplers = []
for episode_arg in range(train_episodes):
    sample = Sinusoid(RNG, batch_size, min_amplitude, max_amplitude)
    train_samplers.append(sample)


tests_samplers = []
for episode_arg in range(test_episodes):
    sample = Sinusoid(RNG, batch_size, min_amplitude, max_amplitude)
    tests_samplers.append(sample)


fit = MAML(meta_model, compute_loss, optimizer, learning_rate)
losses = fit(RNG, train_samplers, epochs)
meta_model.save_weights(weights_filename)

predict = Predict(meta_model, learning_rate, compute_loss)
sampler = tests_samplers[1]
x_support, y_support = sampler(batch_size)
x_queries, y_queries = sampler(100, equally_spaced=True)

steps = [0, 3, 6, 9]
plt.plot(x_support, y_support, '^', label='support points')
plt.plot(x_queries, y_queries, label='true function')
for step in steps:
    y_pred = predict(x_support, y_support, x_queries, step)
    plt.plot(x_queries, y_pred, '--', label=f'{step} transfer steps')
plt.legend()
plt.ylim(-5, 5)
plt.xlim(-6, 6)
plt.show()
