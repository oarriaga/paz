import jax
from keras.utils import PyDataset


class Generator(PyDataset):
    def __init__(self, key, images, detections, batch, **kwargs):
        super().__init__(**kwargs)
        if len(images) != len(detections):
            raise ValueError("Images and detections must have the same length.")
        self.images = images
        self.detections = detections
        self.batch = batch
        self.keys = jax.random.split(
            key, len(images) + 1
        )  # this should be number of batches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, batch_index):
        image = paz.image.load(self.images[batch_index])
        detections = self.detections[batch_index]
        inputs, labels = self.batch(self.keys[batch_index], detections, image)
        return inputs, labels

    def on_epoch_end(self):
        self.keys = jax.random.split(self.keys[-1], len(self.images) + 1)
