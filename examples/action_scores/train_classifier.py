import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import keras

import paz
from paz.datasets import fer
from callbacks import ScalarActionScore, FeatureExtractor


class FERSequence(keras.utils.PyDataset):
    def __init__(self, images, labels, batch_size):
        super().__init__()
        self.images = np.asarray(images, "float32") / 255.0
        self.labels = np.asarray(labels, "float32")
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        return self.images[chunk], self.labels[chunk]


def find_feature_layer(model):
    adds = [layer.name for layer in model.layers
            if layer.__class__.__name__ == "Add"]
    return adds[-1]


def parse_arguments():
    parser = argparse.ArgumentParser(description="FER + action scores")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.root, exist_ok=True)
    train_images, train_labels = fer.load("train")
    valid_images, valid_labels = fer.load("validation")
    model = paz.models.MiniXception((48, 48, 1), len(fer.get_class_names()))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    train = FERSequence(train_images, train_labels, args.batch_size)
    valid = FERSequence(valid_images, valid_labels, args.batch_size)
    scores = ScalarActionScore(train, [keras.losses.categorical_crossentropy],
                               args.epochs, os.path.join(args.root, "scores.hdf5"))  # fmt: skip
    features = FeatureExtractor(find_feature_layer(model), train,
                                os.path.join(args.root, "features.hdf5"))
    log = keras.callbacks.CSVLogger(os.path.join(args.root, "log.csv"))
    model.fit(train, epochs=args.epochs, validation_data=valid,
              callbacks=[log, scores, features])


if __name__ == "__main__":
    main()
