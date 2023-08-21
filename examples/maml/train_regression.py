import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from maml import MAML, MLP, Predict
from cnn import CONVNET
from sinusoid import Sinusoid


seed = 777
meta_model = MLP()
# meta_model = CONVNET(10, (28, 28, 1))
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
