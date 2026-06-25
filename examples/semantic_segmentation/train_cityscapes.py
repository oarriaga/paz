import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import keras

import paz
from paz.datasets import cityscapes


class CityscapesSequence(keras.utils.PyDataset):
    def __init__(self, image_paths, label_paths, num_classes, size, batch_size):
        super().__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.num_classes = num_classes
        self.size = size
        self.batch_size = batch_size
        self.mean = np.array(paz.image.BGR_IMAGENET_MEAN, "float32")

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        chunk = slice(index * self.batch_size, (index + 1) * self.batch_size)
        images, labels = [], []
        for image_path in self.image_paths[chunk]:
            images.append(cityscapes.load_image(image_path, self.size))
        for label_path in self.label_paths[chunk]:
            labels.append(cityscapes.load_mask(label_path, self.size))
        images = np.asarray(images, "float32")[..., ::-1] - self.mean
        targets = np.eye(self.num_classes, dtype="float32")[np.asarray(labels)]
        return images, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Cityscapes segmentation")
    parser.add_argument("--root", required=True, help="Cityscapes directory")
    parser.add_argument("--save_path", default="experiments")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--backbone_weights", default="imagenet")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    H = W = args.image_size
    num_classes = len(cityscapes.get_class_names())
    train_paths = cityscapes.load(args.root, "train")
    valid_paths = cityscapes.load(args.root, "val")
    weights = args.backbone_weights or None
    model = paz.models.UNET_VGG16(num_classes, (H, W, 3), weights,
                                  activation="softmax")
    optimizer = keras.optimizers.Adam(args.learning_rate)
    model.compile(optimizer, loss=paz.losses.dice, metrics=["mse"])
    train = CityscapesSequence(*train_paths, num_classes, (H, W),
                               args.batch_size)
    valid = CityscapesSequence(*valid_paths, num_classes, (H, W),
                               args.batch_size)
    save_weights = os.path.join(args.save_path, "unet_cityscapes.weights.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(args.save_path, "log.csv")),
        keras.callbacks.ModelCheckpoint(save_weights, save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.EarlyStopping("val_loss", patience=5),
        keras.callbacks.ReduceLROnPlateau("val_loss", patience=2),
    ]
    model.fit(train, validation_data=valid, epochs=args.epochs,
              callbacks=callbacks)
