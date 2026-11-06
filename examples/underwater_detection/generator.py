import math

import numpy as np
import jax
import jax.numpy as jp
from keras.utils import PyDataset

import paz


class TrainGenerator(PyDataset):
    def __init__(self, key, image_paths, detections, copy_indices,
                 batch_size, size, workers):
        super().__init__(workers=workers, use_multiprocessing=False)
        self.order_key, self.copy_key = jax.random.split(key)
        self.image_paths, self.detections = image_paths, detections
        self.copy_indices, self.batch_size = copy_indices, batch_size
        self.size, self.max_boxes = size, compute_max_boxes(detections)
        self.epoch = 0

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, batch_arg):
        # reads the epoch at call time, so a resumed run repeats every batch
        epoch_key = jax.random.fold_in(self.order_key, self.epoch)
        positions = self.select_positions(epoch_key, batch_arg)
        batch_key = jax.random.fold_in(epoch_key, batch_arg)
        keys = jax.random.split(batch_key, self.batch_size)
        samples = [self.build_sample(*args) for args in zip(keys, positions)]
        return stack_samples(samples)

    def select_positions(self, epoch_key, batch_arg):
        order = jax.random.permutation(epoch_key, len(self.image_paths))
        lower = batch_arg * self.batch_size
        return np.asarray(order)[lower:lower + self.batch_size].tolist()

    def build_sample(self, key, position):
        image, detections = self.read_sample(position)
        return transform_train(key, image, detections, self.size)

    def read_sample(self, position):
        args = (self.image_paths[position], self.detections[position])
        image, detections = load_sample(*args, self.max_boxes)
        if self.copy_indices[position] == 0:
            sample = (image, detections)
        else:
            copy_key = jax.random.fold_in(self.copy_key, position)
            sample = augment_copy(copy_key, image, detections)
        return sample


class ValidationGenerator(PyDataset):
    def __init__(self, image_paths, detections, batch_size, size, workers):
        super().__init__(workers=workers, use_multiprocessing=False)
        self.image_paths, self.detections = image_paths, detections
        self.batch_size, self.size = batch_size, size
        self.max_boxes = compute_max_boxes(detections)

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, batch_arg):
        lower = batch_arg * self.batch_size
        upper = min(lower + self.batch_size, len(self.image_paths))
        samples = [self.build_sample(arg) for arg in range(lower, upper)]
        return stack_samples(samples)

    def build_sample(self, position):
        args = (self.image_paths[position], self.detections[position])
        image, detections = load_sample(*args, self.max_boxes)
        return transform_validation(image, detections, self.size)


def validate_labels(detections, num_classes):
    labels = np.concatenate([boxes[:, 4] for boxes in detections])
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"Labels must lie in [0, {num_classes})")


def load_sample(path, detections, max_boxes):
    image = paz.image.load(path)
    H, W = paz.image.get_size(image)
    detections = paz.detection.normalize(detections, H, W)
    return image, paz.detection.pad(detections, max_boxes)


@jax.jit
def augment_copy(key, image, detections):
    # colour jitter strength of the reference's offline rare-class copies
    args = (key, image, detections, 0.35)
    return paz.detection.random_flip_enhance_blur(*args)


@paz.partial(jax.jit, static_argnames=("size",))
def transform_train(key, image, detections, size):
    args = (key, image, detections, size)
    image, detections = paz.detection.augment_to_square(*args)
    return standardize(image), to_center_form(detections)


@paz.partial(jax.jit, static_argnames=("size",))
def transform_validation(image, detections, size):
    window = jp.array([0.0, 0.0, 1.0, 1.0])
    image = paz.image.crop_and_resize(image, window, size, size)
    return standardize(image), to_center_form(detections)


def standardize(image):
    # the torchvision ImageNet statistics the weights were trained with
    mean = jp.array([0.485, 0.456, 0.406])
    stdv = jp.array([0.229, 0.224, 0.225])
    return paz.image.standardize(image / 255.0, mean, stdv)


def to_center_form(detections):
    boxes, class_args = paz.detection.split(detections)
    return paz.detection.merge(paz.boxes.to_center_form(boxes), class_args)


def stack_samples(samples):
    images, targets = zip(*samples)
    return np.stack(images), np.stack(targets)


def compute_max_boxes(detections):
    return max(len(boxes) for boxes in detections)
