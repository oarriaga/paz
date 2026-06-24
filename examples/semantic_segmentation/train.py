import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import keras

import paz
from paz.datasets import shapes


class ShapesSequence(keras.utils.PyDataset):
    def __init__(self, images, masks, num_classes, batch_size):
        super().__init__()
        mean = np.array(paz.image.BGR_IMAGENET_MEAN, "float32")
        images = np.asarray(images, "float32")[..., ::-1] - mean
        labels = np.asarray(masks, "int32")[..., 0]
        self.images = images
        self.targets = np.eye(num_classes, dtype="float32")[labels]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        return self.images[chunk], self.targets[chunk]


def parse_arguments():
    parser = argparse.ArgumentParser(description="UNet Shapes segmentation")
    parser.add_argument("--save_path", default="experiments")
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--validation_split", default=0.2, type=float)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.save_path, exist_ok=True)
    H = W = args.image_size
    num_classes = len(shapes.get_class_names()) + 1
    images, _, _, masks = shapes.load(H, W, 0.3, 3, args.num_samples)
    split = int(len(images) * (1 - args.validation_split))
    model = paz.models.UNET_VGG16(num_classes, (H, W, 3), "imagenet",
                                  activation="softmax")
    optimizer = keras.optimizers.Adam(args.learning_rate)
    model.compile(optimizer, loss=paz.losses.dice, metrics=["mse"])
    train = ShapesSequence(images[:split], masks[:split], num_classes,
                           args.batch_size)
    valid = ShapesSequence(images[split:], masks[split:], num_classes,
                           args.batch_size)
    weights = os.path.join(args.save_path, "unet_shapes.weights.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.save_path, "log.csv")),
        keras.callbacks.ModelCheckpoint(weights, save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.EarlyStopping("val_loss", patience=5),
        keras.callbacks.ReduceLROnPlateau("val_loss", patience=2),
    ]
    model.fit(train, validation_data=valid, epochs=args.epochs,
              callbacks=callbacks)


if __name__ == "__main__":
    main()
