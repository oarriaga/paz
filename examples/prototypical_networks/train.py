import os
import argparse
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks as cb

from paz.models import ProtoEmbedding, ProtoNet
from paz.utils import build_directory, write_dictionary, write_weights
from paz.datasets.omniglot import (load, remove_classes, split_data,
                                   sample_between_alphabet,
                                   sample_within_alphabet, Generator)


# TODO move to optimization and add tests
def schedule(period=20, rate=0.5):
    def apply(epoch, learning_rate):
        if ((epoch % period) == 0) and (epoch != 0):
            learning_rate = rate * learning_rate
        return learning_rate
    return apply


description = 'Train and evaluation of prototypical networks'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--root', default='experiments', type=str)
parser.add_argument('--label', default='PROTONET', type=str)
parser.add_argument('--image_H', default=28, type=int)
parser.add_argument('--image_W', default=28, type=int)
parser.add_argument('--num_blocks', default=4, type=int)
parser.add_argument('--steps_per_epoch', default=100, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--period', default=10, type=int)
parser.add_argument('--rate', default=0.5, type=int)
parser.add_argument('--train_classes', default=964, type=int)
parser.add_argument('--validation_split', default=0.20, type=float)
parser.add_argument('--train_path', default='omniglot/images_background/')
parser.add_argument('--tests_path', default='omniglot/images_evaluation/')
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--loss', default='sparse_categorical_crossentropy')
parser.add_argument('--metric', default='sparse_categorical_accuracy')
parser.add_argument('--train_ways', default=60, type=int)
parser.add_argument('--train_shots', default=5, type=int)
parser.add_argument('--train_queries', default=5, type=int)
parser.add_argument('--test_steps', default=1000, type=int)
parser.add_argument('--test_ways', nargs='+', default=[5, 20])
parser.add_argument('--test_shots', nargs='+', default=[1, 5])
parser.add_argument('--test_queries', default=1, type=int)
parser.add_argument('--stop_patience', default=100, type=int)
parser.add_argument('--stop_delta', default=1e-3, type=int)
args = parser.parse_args()


RNG = np.random.default_rng(args.seed)
tf.random.set_seed(args.seed)

directory = build_directory(args.root, args.label)
write_dictionary(args.__dict__, directory, 'parameters.json')

image_shape = (args.image_H, args.image_W, 1)
train_args = (args.train_ways, args.train_shots, args.train_queries)
embed = ProtoEmbedding(image_shape, args.num_blocks)
model = ProtoNet(embed, *train_args, image_shape)
optimizer = Adam(args.learning_rate)
metrics = [args.metric]
model.compile(Adam(args.learning_rate), loss=args.loss, metrics=metrics)

callbacks = [
    cb.LearningRateScheduler(schedule(args.period, args.rate), verbose=1),
    cb.CSVLogger(os.path.join(directory, 'log.csv')),
    cb.EarlyStopping('val_loss', args.stop_delta, args.stop_patience, 1)
]

train_data = load('train', image_shape[:2], True)
train_data = remove_classes(RNG, train_data, args.train_classes)
train_data, validation_data = split_data(train_data, args.validation_split)

sampler = partial(sample_between_alphabet, RNG, train_data, *train_args)
sequence = Generator(sampler, *train_args, image_shape, args.steps_per_epoch)
sampler = partial(sample_between_alphabet, RNG, validation_data, *train_args)
validation_data = Generator(sampler, *train_args, image_shape, 100)

model.fit(sequence,
          epochs=args.epochs,
          callbacks=callbacks,
          validation_data=validation_data)

results = {}
for way in args.test_ways:
    for shot in args.test_shots:
        test_model = ProtoNet(embed, way, shot, args.test_queries, image_shape)
        test_model.compile(optimizer, loss=args.loss, metrics=metrics)
        test_args = (way, shot, args.test_queries)

        data = load('test', image_shape[:2], flat=False)
        sampler = partial(sample_within_alphabet, RNG, data, *test_args)
        sequence = Generator(sampler, *test_args, image_shape, args.test_steps)
        losses, accuracy = test_model.evaluate(sequence)
        accuracy = round(100 * accuracy, 2)
        results[f'{way}-way_{shot}-shot_within_alphabet'] = accuracy
        print(f'Within alphabet {way}-way {shot}-shot accuracy {accuracy} %')

        data = load('test', image_shape[:2], flat=True)
        sampler = partial(sample_between_alphabet, RNG, data, *test_args)
        sequence = Generator(sampler, *test_args, image_shape, args.test_steps)
        losses, accuracy = test_model.evaluate(sequence)
        accuracy = round(100 * accuracy, 2)
        results[f'{way}-way_{shot}-shot_between_alphabet'] = accuracy
        print(f'Between alphabet {way}-way {shot}-shot accuracy {accuracy} %')

write_weights(embed, directory)
write_dictionary(results, directory, 'test_accuracies.json')
