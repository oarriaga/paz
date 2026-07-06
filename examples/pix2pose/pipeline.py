import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jp
import keras

import paz


def render_views(render_image, render_coordinates, poses):
    images, coordinates, masks = [], [], []
    for pose in poses:
        nocs, mask = render_coordinates(pose)
        images.append(np.asarray(render_image(pose)))
        coordinates.append(np.asarray(nocs))
        masks.append(np.asarray(mask))
    images = np.stack(images).astype("uint8")
    coordinates = np.stack(coordinates).astype("float32")
    masks = np.stack(masks).astype("float32")
    return images, coordinates, masks


def build_label(coordinates, mask):
    return np.concatenate([coordinates, mask[..., None]], axis=-1)


def load_backgrounds(path, image_size, num_backgrounds=None):
    if path is None:
        return None
    files = sorted(os.path.join(path, name) for name in os.listdir(path))
    files = files if num_backgrounds is None else files[:num_backgrounds]
    size = (image_size, image_size)
    crops = [paz.image.resize_opencv(paz.image.load(name), size)
             for name in files]
    return np.stack(crops).astype("uint8")


class Pix2PoseSequence(keras.utils.PyDataset):
    def __init__(self, images, coordinates, masks, batch_size,
                 backgrounds=None, num_occlusions=1, seed=0):
        super().__init__()
        self.images = images
        self.labels = build_label(coordinates, masks)
        self.masks = masks
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(seed)
        backgrounds = None if backgrounds is None else jp.asarray(backgrounds)
        randomize = partial(paz.image.randomize_rendered_image,
                            backgrounds=backgrounds,
                            num_occlusions=num_occlusions)
        self.randomize = jax.jit(jax.vmap(randomize, in_axes=(0, 0, 0)))

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        stop = start + self.batch_size
        self.key, batch_key = jax.random.split(self.key)
        keys = jax.random.split(batch_key, self.batch_size)
        images = self.images[start:stop]
        masks = self.masks[start:stop]
        inputs = self.randomize(keys, images, masks)
        return np.asarray(inputs) / 255.0, self.labels[start:stop]
