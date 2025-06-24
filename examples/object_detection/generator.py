import math
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
        self.images = images
        self.labels = labels
        self.pipeline = pipeline
        self.batch_size = batch_size
        num_batches = compute_num_batches(self.images, batch_size)
        self.keys = jax.random.split(key, num_batches + 1)

    def __len__(self):
        return compute_num_batches(self.images, self.batch_size)

    def __getitem__(self, batch_arg):
        lower_arg = batch_arg * self.batch_size
        upper_arg = min(lower_arg + self.batch_size, len(self.images))
        images = self.images[lower_arg:upper_arg]
        labels = self.labels[lower_arg:upper_arg]
        return self.pipeline(self.keys[batch_arg], images, labels)

    def on_epoch_end(self):
        # repopulates keys for the next epoch
        self.keys = jax.random.split(self.keys[-1], len(self.images) + 1)
