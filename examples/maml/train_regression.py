import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from paz.utils import build_directory, write_dictionary, write_weights

from maml import MLP, MAML, Predict
from sinusoid import Sinusoid

description = 'Train and evaluation of model agnostic meta learning (MAML)'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--label', default='MAML-SINUSOID', type=str)
parser.add_argument('--meta_learning_rate', default=0.001, type=float)
parser.add_argument('--task_learning_rate', default=0.01, type=float)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--train_steps', default=10_000, type=int)
parser.add_argument('--train_ways', default=5, type=int)
parser.add_argument('--train_shots', default=5, type=int)
parser.add_argument('--train_queries', default=5, type=int)
parser.add_argument('--hidden_size', default=40, type=int)

parser.add_argument('--min_amplitude', default=0.1, type=float)
parser.add_argument('--max_amplitude', default=5.0, type=float)
parser.add_argument('--min_x', default=-5.0, type=float)
parser.add_argument('--max_x', default=5.0, type=float)
args = parser.parse_args()

directory = build_directory(args.root, args.label)
write_dictionary(args.__dict__, directory, 'parameters.json')

RNG = np.random.default_rng(args.seed)
meta_model = MLP(args.hidden_size)
compute_loss = MeanSquaredError()
optimizer = Adam(learning_rate=args.meta_learning_rate)

sample = Sinusoid(RNG, args.batch_size, args.min_amplitude, args.max_amplitude)
fit = MAML(meta_model, compute_loss, optimizer, args.task_learning_rate)
losses = fit(RNG, sample, args.train_steps)
write_weights(meta_model, directory)

predict = Predict(meta_model, args.task_learning_rate, compute_loss)
x_support, y_support = sample(args.batch_size)[0]
x_queries, y_queries = sample(100, equally_spaced=True)[1]

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
