import math
import numpy as np
from keras.utils import PyDataset
import jax


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
