import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import jax
import jax.numpy as jp
import keras

import paz


def augment_face(key, image):
    key_flip, key_light = jax.random.split(key)
    image = paz.image.random_flip_left_right(key_flip, image)
    return paz.image.random_brightness(key_light, image)


class IMDBSequence(keras.utils.PyDataset):
    def __init__(self, images, labels, batch_size, augment=False, seed=0):
        super().__init__()
        self.images = np.asarray(images, "uint8")
        self.labels = np.asarray(labels, "float32")
        self.batch_size = batch_size
        self.augment = augment
        self.key = jax.random.PRNGKey(seed)
        self.augment_images = jax.jit(jax.vmap(augment_face))

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        images = self.images[chunk]
        if self.augment:
            key = jax.random.fold_in(self.key, index)
            keys = jax.random.split(key, len(images))
            images = self.augment_images(keys, jp.asarray(images))
        return np.asarray(images, "float32") / 255.0, self.labels[chunk]


def load_arrays(data, split):
    images = np.load(os.path.join(data, f"{split}_images.npy"))
    labels = np.load(os.path.join(data, f"{split}_labels.npy"))
    return images, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniXception IMDB training")
    parser.add_argument("--data", default="data")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)
    train_images, train_labels = load_arrays(args.data, "train")
    valid_images, valid_labels = load_arrays(args.data, "validation")
    num_classes = len(paz.datasets.labels("IMDB"))
    model = paz.models.build_mini_xception_imdb((64, 64, 1), num_classes)
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
