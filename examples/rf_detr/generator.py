"""Batches detection data into RF-DETR inputs and set-prediction targets.

Images are squeezed into the detector's square input and normalized with the
torchvision ImageNet statistics the weights were trained with. Targets are
normalized ``(cx, cy, w, h)`` plus a class index, padded to a fixed count with
-1 so the loss keeps static shapes.

Augmentation is a horizontal flip only. The colour helpers in ``paz.image``
expect 0-255 images and the scale-jitter ones move boxes out of normalized
coordinates, so neither composes with this pipeline as written.
"""
import math

import numpy as np
import jax
import jax.numpy as jp
from keras.utils import PyDataset

import paz

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STDV = (0.229, 0.224, 0.225)


class Generator(PyDataset):
    def __init__(self, key, images, labels, batch_size, pipeline, workers):
        super().__init__(workers=workers, use_multiprocessing=False)
        if len(images) != len(labels):
            raise ValueError("Images and labels must have same length.")
        self.images, self.labels = list(images), list(labels)
        self.pipeline, self.batch_size = pipeline, batch_size
        self.batches_per_epoch = math.ceil(len(images) / batch_size)
        self.master_key = key
        self.reset()

    def reset(self):
        keys = jax.random.split(self.master_key, 3)
        self.master_key, batch_key, shuffle_key = keys
        self.keys = jax.random.split(batch_key, self.batches_per_epoch + 1)
        order = jax.random.permutation(shuffle_key, len(self.images))
        order = np.asarray(order)
        self.images = [self.images[arg] for arg in order]
        self.labels = [self.labels[arg] for arg in order]

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, batch_arg):
        lower = batch_arg * self.batch_size
        upper = min(lower + self.batch_size, len(self.images))
        images = self.images[lower:upper]
        labels = self.labels[lower:upper]
        return self.pipeline(self.keys[batch_arg], images, labels)

    def on_epoch_end(self):
        self.reset()


def preprocess_batch(key, images, detections, resolution, max_boxes, augment):
    images, detections = load_batch(images, detections, resolution, max_boxes)
    keys = jax.random.split(key, len(images))
    args = keys, jp.asarray(images, jp.float32), jp.asarray(detections)
    images, detections = transform_batch(*args, augment)
    return np.asarray(images, "float32"), np.asarray(detections, "float32")


def load_batch(paths, detections, resolution, max_boxes):
    """CPU I/O boundary: read each image, resize it and pad its boxes."""
    images, labels = [], []
    for path, detection in zip(paths, detections):
        image = paz.image.load(path)
        height, width = paz.image.get_size(image)
        size = (resolution, resolution)
        resized = paz.image.resize(paz.cast(image, "float32"), size)
        images.append(np.asarray(resized))
        detection = np.asarray(detection, "float32")
        scale = np.array([1.0 / width, 1.0 / height] * 2)
        boxes = detection[:, :4] * scale
        label = np.concatenate([boxes, detection[:, 4:]], axis=1)[:max_boxes]
        padding = ((0, max_boxes - len(label)), (0, 0))
        labels.append(np.pad(label, padding, constant_values=-1))
    return np.stack(images), np.stack(labels)


@paz.partial(jax.jit, static_argnames=("augment",))
def transform_batch(keys, images, detections, augment):
    mean = jp.asarray(IMAGENET_MEAN, jp.float32)
    stdv = jp.asarray(IMAGENET_STDV, jp.float32)

    def transform(key, image, detection):
        if augment:
            image, detection = paz.detection.random_flip(key, image, detection)
        image = paz.image.standardize(image / 255.0, mean, stdv)
        return image, to_center_form(detection)

    return jax.vmap(transform)(keys, images, detections)


def to_center_form(detection):
    """Keeps the -1 padding rows recognizable by their negative class."""
    boxes = paz.boxes.to_center_form(detection[:, :4])
    return jp.concatenate([boxes, detection[:, 4:]], axis=1)
