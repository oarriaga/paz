import os
import argparse
from functools import partial

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import numpy as np
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, CSVLogger, EarlyStopping
import paz
from paz.models.classification import ProtoEmbedding, ProtoNet
from paz.datasets.fsclvr import (
    load,
    sample,
    split_data,
    remove_classes,
)


class Generator(Sequence):
    def __init__(
        self,
        sampler,
        num_classes,
        num_support,
        num_queries,
        image_shape,
        num_steps=2000,
    ):
        self.sampler = sampler
        self.support_shape = (num_classes, num_support, *image_shape)
        self.queries_shape = (num_classes, num_queries, *image_shape)
        self.num_steps = num_steps

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        (support, support_labels), (queries, queries_labels) = self.sampler()
        support = np.reshape(support, self.support_shape)
        queries = np.reshape(queries, self.queries_shape)
        return {"support": support, "queries": queries}, queries_labels


# TODO move to optimization and add tests
def schedule(period=20, rate=0.5):
    def apply(epoch, learning_rate):
        if ((epoch % period) == 0) and (epoch != 0):
            learning_rate = rate * learning_rate
        return learning_rate

    return apply


description = "Train and evaluate prototypical networks"
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--label", default="PROTONET", type=str)
parser.add_argument("--image_H", default=480 // 3, type=int)
parser.add_argument("--image_W", default=640 // 3, type=int)
parser.add_argument("--num_channels", default=3, type=int)
parser.add_argument("--num_blocks", default=4, type=int)
parser.add_argument("--steps_per_epoch", default=100, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--period", default=10, type=int)
parser.add_argument("--rate", default=0.5, type=int)
parser.add_argument("--train_classes", default=30, type=int)
parser.add_argument("--validation_split", default=0.20, type=float)
parser.add_argument("--train_path", default="omniglot/images_background/")
parser.add_argument("--tests_path", default="omniglot/images_evaluation/")
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--loss", default="sparse_categorical_crossentropy")
parser.add_argument("--metric", default="sparse_categorical_accuracy")
parser.add_argument("--train_ways", default=5, type=int)
parser.add_argument("--train_shots", default=5, type=int)
parser.add_argument("--train_queries", default=1, type=int)
parser.add_argument("--test_steps", default=1000, type=int)
parser.add_argument("--test_ways", nargs="+", default=[5, 20])
parser.add_argument("--test_shots", nargs="+", default=[1, 5])
parser.add_argument("--test_queries", default=1, type=int)
parser.add_argument("--stop_patience", default=20, type=int)
parser.add_argument("--stop_delta", default=1e-3, type=int)
parser.add_argument("--scene", default="plain", type=str)
args = parser.parse_args()
directory, _ = paz.logger.setup(args)
RNG = np.random.default_rng(args.seed)

image_shape = (args.image_H, args.image_W, args.num_channels)
train_args = (args.train_ways, args.train_shots, args.train_queries)
valid_args = (args.train_ways, args.train_shots, args.train_queries)
embed = ProtoEmbedding(image_shape, args.num_blocks)
model = ProtoNet(embed, *train_args, image_shape)
optimizer = Adam(args.learning_rate)
metrics = [args.metric]
model.compile(optimizer, loss=args.loss, metrics=metrics, jit_compile=True)

callbacks = [
    LearningRateScheduler(schedule(args.period, args.rate), verbose=1),
    CSVLogger(os.path.join(directory, "log.csv")),
    EarlyStopping("val_loss", args.stop_delta, args.stop_patience, 1),
]

train_data = load(args.scene, "train", image_shape[:2])
train_data = remove_classes(RNG, train_data, args.train_classes)
train_data, validation_data = split_data(train_data, args.validation_split)

sampler = partial(sample, RNG, train_data, *train_args)
sequence = Generator(sampler, *train_args, image_shape, args.steps_per_epoch)
sampler = partial(sample, RNG, validation_data, *valid_args)
if len(validation_data) == 0:
    validation_data = None
else:
    validation_data = Generator(sampler, *valid_args, image_shape, 100)

model.fit(
    sequence,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=validation_data,
)

results = {}
for way in args.test_ways:
    for shot in args.test_shots:
        test_model = ProtoNet(embed, way, shot, args.test_queries, image_shape)
        test_model.compile(optimizer, loss=args.loss, metrics=metrics)
        test_args = (way, shot, args.test_queries)

        data = load(args.scene, "test", image_shape[:2])
        sampler = partial(sample, RNG, data, *test_args)
        sequence = Generator(sampler, *test_args, image_shape, args.test_steps)
        losses, accuracy = test_model.evaluate(sequence)
        accuracy = round(100 * accuracy, 2)
        results[f"{way}-way_{shot}-shot_within_alphabet"] = accuracy
        print(f"{way}-way {shot}-shot accuracy {accuracy} %")

paz.logger.write_weights(test_model, directory)
test_model.save(os.path.join(directory, f"{test_model.name}.keras"))
paz.logger.write_dictionary(results, os.path.join(directory, "results.json"))
