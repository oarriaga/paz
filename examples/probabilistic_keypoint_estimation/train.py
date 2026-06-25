import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import jax
import jax.numpy as jp
import keras

import paz
import facial_keypoints


class KeypointSequence(keras.utils.PyDataset):
    def __init__(self, images, keypoints, batch_size, augment=False, seed=0):
        super().__init__()
        self.images = np.asarray(images, "uint8")
        self.keypoints = paz.gaussian_mixture.normalize_points(
            np.asarray(keypoints, "float32"), 96, 96)
        self.batch_size = batch_size
        self.augment = augment
        self.key = jax.random.PRNGKey(seed)
        self.augment_images = jax.jit(jax.vmap(paz.image.random_brightness))

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        images = self.images[chunk][..., None]
        if self.augment:
            key = jax.random.fold_in(self.key, index)
            keys = jax.random.split(key, len(images))
            images = self.augment_images(keys, jp.asarray(images))
        images = np.asarray(images, "float32") / 255.0
        return images, np.asarray(self.keypoints[chunk])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probabilistic keypoints")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--save_path", default="experiments")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--validation_split", default=0.2, type=float)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    images, keypoints = facial_keypoints.load(args.root, "train")
    split = int(len(images) * (1 - args.validation_split))
    model = paz.models.GaussianMixtureModel((96, 96, 1), 15, 8)
    model.summary()
    optimizer = keras.optimizers.Adam(args.learning_rate, amsgrad=True)
    model.compile(optimizer, loss=paz.losses.gaussian_mixture_nll)
    train = KeypointSequence(images[:split], keypoints[:split],
                             args.batch_size, True)
    valid = KeypointSequence(images[split:], keypoints[split:],
                             args.batch_size)
    weights = os.path.join(args.save_path, "gaussian_mixture.weights.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.save_path, "log.csv")),
        keras.callbacks.ModelCheckpoint(weights, save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.EarlyStopping("val_loss", patience=7),
        keras.callbacks.ReduceLROnPlateau("val_loss", patience=3),
    ]
    model.fit(train, validation_data=valid, epochs=args.epochs,
              callbacks=callbacks)
