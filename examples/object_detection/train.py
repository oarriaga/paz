import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import jax.numpy as jp
import jax
import paz
import keras
from generator import Generator
from pipeline import AugmentDetection

key = jax.random.PRNGKey(777)
batch_size = 32
H = 300
W = 300
num_classes = 20
max_num_boxes = 25
prior_boxes = paz.models.detection.utils.create_prior_boxes("VOC")
variances = [0.1, 0.1, 0.2, 0.2]
match_IOU = 0.5
mean = jp.array(paz.image.BGR_IMAGENET_MEAN)

images_07, class_args_07, boxes_07 = paz.datasets.load("VOC2007", "trainval")
images_12, class_args_12, boxes_12 = paz.datasets.load("VOC2012", "trainval")
train_images = images_07 + images_12
train_class_args = class_args_07 + class_args_12
train_boxes = boxes_07 + boxes_12
train_data = (train_images, train_boxes, train_class_args)

test_images, test_class_args, test_boxes = paz.datasets.load("VOC2007", "test")

model = paz.models.SSD300(
    num_classes + 1, base_weights="VGG", head_weights=None, trainable_base=False
)
model.summary()

metrics = {
    "boxes": [
        paz.losses.multibox.regression,
        paz.losses.multibox.positive_classification,
        paz.losses.multibox.negative_classification,
    ]
}

optimizer = keras.optimizers.SGD(0.001, 0.9)
model.compile(optimizer, loss=paz.losses.multibox.call, metrics=metrics)
pipeline = paz.lock(
    AugmentDetection,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
)

generator = Generator(key, *train_data, batch_size, pipeline)
model.fit(generator, epochs=10)
