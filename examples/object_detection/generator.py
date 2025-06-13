import math
from keras.utils import PyDataset
import jax


def compute_num_batches(samples, batch_size):
    return math.ceil(len(samples) / batch_size)


class Generator(PyDataset):
    def __init__(
        self, key, images, boxes, class_args, batch_size, pipeline, **kwargs
    ):
        super().__init__(**kwargs)
        if len(images) != len(boxes) != len(class_args):
            raise ValueError("Images, boxes and classes must have same length.")
        self.images = images
        self.boxes = boxes
        self.class_args = class_args
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
        boxes = self.boxes[lower_arg:upper_arg]
        class_args = self.class_args[lower_arg:upper_arg]
        return self.pipeline(self.keys[batch_arg], images, boxes, class_args)

    def on_epoch_end(self):
        # repopulates keys for the next epoch
        self.keys = jax.random.split(self.keys[-1], len(self.images) + 1)
