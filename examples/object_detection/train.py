import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import paz
import keras

images_07, class_args_07, boxes_07 = paz.datasets.load("VOC2007", "trainval")
images_12, class_args_12, boxes_12 = paz.datasets.load("VOC2012", "trainval")
train_images = images_07 + images_12
train_class_args = class_args_07 + class_args_12
train_boxes = boxes_07 + boxes_12

test_images, test_class_args, test_boxes = paz.datasets.load("VOC2007", "test")


model = paz.models.SSD300(20, base_weights="VGG", head_weights=None)
model.summary()

metrics = {
    "boxes": [
        paz.losses.multibox.localization,
        paz.losses.multibox.positive_classification,
        paz.losses.multibox.negative_classification,
    ]
}


optimizer = keras.optimizers.SGD(0.001, 0.9)
model.compile(optimizer, paz.losses.multibox.call, metrics)
