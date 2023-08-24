import argparse
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from paz.utils import build_directory, write_dictionary, write_weights
from paz.datasets.omniglot import load, sample_between_alphabet

from maml import CONVNET, MAML, Predict, compute_accuracy

description = 'Train and evaluation of model agnostic meta learning (MAML)'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--label', default='MAML', type=str)
parser.add_argument('--image_shape', nargs='+', default=[28, 28, 1])
parser.add_argument('--num_blocks', default=4, type=int)
parser.add_argument('--meta_learning_rate', default=0.001, type=float)
parser.add_argument('--task_learning_rate', default=0.4, type=float)
parser.add_argument('--train_steps', default=500, type=int)
parser.add_argument('--train_ways', default=5, type=int)
parser.add_argument('--train_shots', default=5, type=int)
parser.add_argument('--train_queries', default=5, type=int)
parser.add_argument('--test_steps', default=100, type=int)
# parser.add_argument('--test_ways', nargs='+', default=[5, 20])
# parser.add_argument('--test_shots', nargs='+', default=[1, 5])
# parser.add_argument('--test_queries', default=1, type=int)
args = parser.parse_args()

directory = build_directory(args.root, args.label)
write_dictionary(args.__dict__, directory, 'parameters.json')

RNG = np.random.default_rng(args.seed)
tf.random.set_seed(args.seed)

# image_shape = (args.image_H, args.image_W, args.image_C)
compute_loss = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=args.meta_learning_rate)
train_args = (args.train_ways, args.train_shots, args.train_queries)

meta_model = CONVNET(args.train_ways, args.image_shape, args.num_blocks)

train_data = load('train', args.image_shape[:2], True)

train_sampler = partial(sample_between_alphabet, RNG, train_data, *train_args)
(x1, y1), (x2, y2) = train_sampler()

fit = MAML(meta_model, compute_loss, optimizer, args.task_learning_rate)
losses = fit(RNG, train_sampler, args.train_steps)
write_weights(meta_model, directory)

tests_data = load('test', args.image_shape[:2], True)
tests_sampler = partial(sample_between_alphabet, RNG, tests_data, *train_args)

predict = Predict(meta_model, args.task_learning_rate, compute_loss)
accuracies = []
for arg in range(args.test_steps):
    ((x_true_support, y_true_support),
     (x_true_queries, y_true_queries)) = tests_sampler()
    y_pred_queries = predict(x_true_support, y_true_support, x_true_queries, 3)
    y_pred_queries = np.argmax(y_pred_queries, axis=1)
    accuracy = compute_accuracy(y_true_queries, y_pred_queries)
    accuracies.append(accuracy)
accuracies = np.array(accuracies)
mean_accuracy = np.mean(accuracies)
print(f'mean accuracy {mean_accuracy}')
results = {}
results[f'{args.train_ways}-way_{args.train_shots}-shot_between_alphabet'] = mean_accuracy
write_dictionary(results, directory, 'test_accuracies.json')
