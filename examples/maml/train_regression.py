import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from maml import MLP, Predict  # MAML
from cnn import MAML
from sinusoid import Sinusoid


seed = 777
meta_model = MLP()
compute_loss = MeanSquaredError()
optimizer = Adam()
learning_rate = 0.01
RNG = np.random.default_rng(seed)
train_steps = 10_000
batch_size = 10
min_amplitude = 0.1
max_amplitude = 5.0
min_x = -5.0
max_x = 5.0
weights_filename = f'MAML_MLP_epochs-{train_steps}.hdf5'


sampler = Sinusoid(RNG, batch_size, min_amplitude, max_amplitude)
fit = MAML(meta_model, compute_loss, optimizer, learning_rate)
losses = fit(RNG, sampler, train_steps)
meta_model.save_weights(weights_filename)

predict = Predict(meta_model, learning_rate, compute_loss)
x_support, y_support = sampler(batch_size)[0]
x_queries, y_queries = sampler(100, equally_spaced=True)[1]

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
