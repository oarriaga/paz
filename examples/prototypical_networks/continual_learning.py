import os

os.environ["KERAS_BACKEND"] = "jax"
from functools import partial
import argparse
import jax
import jax.numpy as jp
import numpy as np

from keras.optimizers import Adam

import omniglot
from protonet import ProtoNet, ProtoEmbedding


def sample_keys(key, dictionary, num_samples):
    keys = list(dictionary.keys())
    args = jp.arange(len(keys))
    args = jax.random.choice(key, args, shape=(num_samples,), replace=False)
    return [keys[arg] for arg in args]


def keys_to_dict(keys, dictionary):
    new_dictionary = {}
    for key in keys:
        if key in dictionary:
            new_dictionary[key] = dictionary[key]
    return new_dictionary


def split_dict(key, dictionary, split_ratio=0.5):
    num_keys = int(split_ratio * len(dictionary))
    keys_split_A = sample_keys(key, dictionary, num_keys)
    keys_split_B = list(set(dictionary.keys()).difference(set(keys_split_A)))
    dictionary_A = keys_to_dict(keys_split_A, dictionary)
    dictionary_B = keys_to_dict(keys_split_B, dictionary)
    return dictionary_A, dictionary_B


description = "Train prototypical networks for continual learning"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--image_H", default=28, type=int)
parser.add_argument("--image_W", default=28, type=int)
parser.add_argument("--num_blocks", default=4, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--pretrain_split", default=0.5, type=float)
parser.add_argument("--train_ways", default=60, type=int)
parser.add_argument("--train_shots", default=5, type=int)
parser.add_argument("--train_queries", default=5, type=int)
parser.add_argument("--metric", default="sparse_categorical_accuracy")
parser.add_argument("--steps_per_epoch", default=100, type=int)
parser.add_argument("--loss", default="sparse_categorical_crossentropy")
args = parser.parse_args()

RNG = np.random.default_rng(args.seed)

languages = omniglot.load("train", flat=False)
pretraining_split = 0.5
key = jax.random.key(args.seed)

few_shot_data, continual_data = split_dict(key, languages, args.pretrain_split)
few_shot_data = omniglot.flatten(few_shot_data)


image_shape = (args.image_H, args.image_W, 1)
train_args = (args.train_ways, args.train_shots, args.train_queries)

embed = ProtoEmbedding(image_shape, args.num_blocks)
model = ProtoNet(embed, *train_args, image_shape)
optimizer = Adam(args.learning_rate)
metrics = [args.metric]
model.compile(Adam(args.learning_rate), loss=args.loss, metrics=metrics)


sampler = partial(
    omniglot.sample_between_alphabet, RNG, few_shot_data, *train_args
)
sequence = omniglot.Generator(
    sampler, *train_args, image_shape, args.steps_per_epoch
)


# each task contains num_continual_classes
# thus the flat dataset must be split into tasks
# we iterate over task, 
support, queries
task_0 = [(x_train_task_0, y_train_task_0), (x_test_task_0, y_test_task_0)]
task_1 = [(x_train_task_1, y_train_task_1), (x_test_task_1, y_test_task_1)]
task_2 = [(x_train_task_2, y_train_task_2), (x_test_task_2, y_test_task_2)]
tasks = [task_0, task_1, task_2]
