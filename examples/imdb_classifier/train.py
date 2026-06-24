import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import keras

import paz


class IMDBSequence(keras.utils.PyDataset):
    def __init__(self, images, labels, batch_size, augment=False):
        super().__init__()
        self.images = np.asarray(images, "float32") / 255.0
        self.labels = np.asarray(labels, "float32")
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        images, labels = self.images[chunk].copy(), self.labels[chunk]
        return augment_batch(images) if self.augment else images, labels


def augment_batch(images):
    flip = np.random.random(len(images)) < 0.5
    images[flip] = images[flip, :, ::-1, :]
    brightness = np.random.uniform(0.9, 1.1, (len(images), 1, 1, 1))
    return np.clip(images * brightness, 0.0, 1.0)


def load_arrays(data, split):
    images = np.load(os.path.join(data, f"{split}_images.npy"))
    labels = np.load(os.path.join(data, f"{split}_labels.npy"))
    return images, labels


def parse_arguments():
    parser = argparse.ArgumentParser(description="MiniXception IMDB training")
    parser.add_argument("--data", default="data")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.root, exist_ok=True)
    train_images, train_labels = load_arrays(args.data, "train")
    valid_images, valid_labels = load_arrays(args.data, "validation")
    num_classes = len(paz.datasets.labels("IMDB"))
    model = paz.models.MiniXception((48, 48, 1), num_classes)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    train = IMDBSequence(train_images, train_labels, args.batch_size, True)
    valid = IMDBSequence(valid_images, valid_labels, args.batch_size)
    weights = os.path.join(args.root, "imdb_mini_XCEPTION_paz_jax.weights.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.root, "log.csv")),
        keras.callbacks.ModelCheckpoint(weights, save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.EarlyStopping("val_loss", patience=5),
        keras.callbacks.ReduceLROnPlateau("val_loss", patience=2),
    ]
    model.fit(train, validation_data=valid, epochs=args.epochs,
              callbacks=callbacks)


if __name__ == "__main__":
    main()
