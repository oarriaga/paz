import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from functools import partial

import numpy as np
import jax
import jax.numpy as jp
import keras

import paz
import facial_keypoints


def augment_image_and_keypoints(key, image, keypoints, rotation_range,
                                delta_scale):
    rotate_key, translate_key, bright_key = jax.random.split(key, 3)
    image, keypoints = paz.keypoints.rotate_image_and_keypoints(
        rotate_key, image, keypoints, rotation_range)
    image, keypoints = paz.keypoints.translate_image_and_keypoints(
        translate_key, image, keypoints, delta_scale)
    image = paz.image.random_brightness(bright_key, image)
    return image, keypoints


class KeypointSequence(keras.utils.PyDataset):
    def __init__(self, images, keypoints, batch_size, augment=False,
                 rotation_range=jp.pi / 12, delta_scale=(0.1, 0.1), seed=0):
        super().__init__()
        self.images = np.asarray(images, "uint8")
        self.keypoints = np.asarray(keypoints, "float32")
        self.batch_size = batch_size
        self.augment = augment
        self.key = jax.random.PRNGKey(seed)
        augment_one = partial(augment_image_and_keypoints,
                              rotation_range=rotation_range,
                              delta_scale=delta_scale)
        self.augment_batch = jax.jit(jax.vmap(augment_one))

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        images = jp.asarray(self.images[chunk][..., None], "float32")
        keypoints = jp.asarray(self.keypoints[chunk])
        if self.augment:
            self.key, batch_key = jax.random.split(self.key)
            keys = jax.random.split(batch_key, len(images))
            images, keypoints = self.augment_batch(keys, images, keypoints)
        images = np.asarray(images) / 255.0
        keypoints = paz.gaussian_mixture.normalize_points(keypoints, 96, 96)
        return images, np.asarray(keypoints)


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
