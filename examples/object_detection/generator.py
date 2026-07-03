import math

import numpy as np
import jax
import jax.numpy as jp
from keras.utils import PyDataset
import paz


def compute_num_batches(samples, batch_size):
    return math.ceil(len(samples) / batch_size)


class Generator(PyDataset):
    def __init__(
        self, key, images, labels, batch_size, pipeline, workers, max_queue_size
    ):
        super().__init__(
            workers=workers,
            use_multiprocessing=False,
            max_queue_size=max_queue_size,
        )
        if len(images) != len(labels):
            raise ValueError("Images and labels must have same length.")
        self.images = list(images)
        self.labels = list(labels)
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.batches_per_epoch = compute_num_batches(self.images, batch_size)
        self.master_key = key
        self.reset()

    def reset(self):
        # Advance the master key, then derive fresh per-batch augmentation keys
        # and a data shuffle for the epoch. Shuffling matters: VOC2007+2012 are
        # concatenated, so without it every batch is class/dataset-correlated,
        # which destabilizes SGD with momentum.
        self.master_key, batch_key, shuffle_key = jax.random.split(
            self.master_key, 3
        )
        self.keys = jax.random.split(batch_key, self.batches_per_epoch + 1)
        order = np.asarray(jax.random.permutation(shuffle_key, len(self.images)))
        self.images = [self.images[arg] for arg in order]
        self.labels = [self.labels[arg] for arg in order]

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, batch_arg):
        lower_arg = batch_arg * self.batch_size
        upper_arg = min(lower_arg + self.batch_size, len(self.images))
        images = self.images[lower_arg:upper_arg]
        labels = self.labels[lower_arg:upper_arg]
        return self.pipeline(self.keys[batch_arg], images, labels)

    def on_epoch_end(self):
        self.reset()


def preprocess_batch(key, images, detections, H, W, prior_boxes, num_classes,
                     match_IOU, variances, mean, max_num_boxes, augment=True):
    images, detections = load_batch(images, detections, H, W, max_num_boxes)
    images = jp.asarray(images, jp.float32)
    detections = jp.asarray(detections, jp.float32)
    keys = jax.random.split(key, len(images))
    mean = jp.asarray(mean, jp.float32)
    images, detections = transform_batch(
        keys, images, detections, prior_boxes, mean, num_classes, match_IOU,
        tuple(variances), augment)
    return np.asarray(images, "float32"), np.asarray(detections, "float32")


def load_batch(paths, detections, H, W, max_boxes):
    """CPU I/O boundary: read and resize each image, scale boxes to H x W
    pixels and pad to a fixed count. Everything downstream is JAX on device."""
    images, labels = [], []
    for path, detection in zip(paths, detections):
        image = paz.image.load(path)
        H_now, W_now = paz.image.get_size(image)
        images.append(np.asarray(paz.image.resize(image, (H, W))))
        detection = np.asarray(detection, "float32")
        scale = np.array([W / W_now, H / H_now, W / W_now, H / H_now])
        boxes = detection[:, :4] * scale
        label = np.concatenate([boxes, detection[:, 4:]], axis=1)[:max_boxes]
        padding = ((0, max_boxes - len(label)), (0, 0))
        labels.append(np.pad(label, padding, constant_values=-1))
    return np.stack(images), np.stack(labels)


# One jitted vmap over the batch. JAX compiles and caches a variant per static
# configuration, so no hand-managed pipeline cache is needed.
@paz.partial(jax.jit, static_argnames=("num_classes", "match_IOU",
                                       "variances", "augment"))
def transform_batch(keys, images, detections, prior_boxes, mean, num_classes,
                    match_IOU, variances, augment):
    def transform(key, image, detection):
        H, W = paz.image.get_size(image)
        detection = paz.detection.normalize(detection, H, W)
        if augment:
            image, detection = paz.detection.augment_detection(
                key, image, detection, mean)
        image = preprocess_image(image, mean)
        detection = paz.detection.encode_detection(
            detection, prior_boxes, num_classes, match_IOU, variances)
        return image, detection

    return jax.vmap(transform)(keys, images, detections)


def preprocess_image(image, mean):
    return paz.image.subtract_mean(paz.image.RGB_to_BGR(image), mean)
