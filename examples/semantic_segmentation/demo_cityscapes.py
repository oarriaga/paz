import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse

import numpy as np
import jax.numpy as jp

import paz
from paz.datasets import cityscapes

parser = argparse.ArgumentParser(description="UNet Cityscapes demo")
parser.add_argument("--image", required=True, help="leftImg8bit RGB image")
parser.add_argument("--weights",
                    default="experiments/unet_cityscapes.weights.h5")
parser.add_argument("--image_size", default=256, type=int)
parser.add_argument("--output", default="unet_cityscapes.jpg")
args = parser.parse_args()

H = W = args.image_size
num_classes = len(cityscapes.get_class_names())
args_model = (num_classes, (H, W, 3), None)
model = paz.models.UNET_VGG16(*args_model, activation="softmax")
model.load_weights(args.weights)
colors = paz.draw.lincolor(num_classes)
mean = jp.array(paz.image.BGR_IMAGENET_MEAN)


def preprocess(image):
    image = paz.image.resize_opencv(image, (H, W))
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    return jp.expand_dims(paz.cast(image, "float32"), axis=0)


image = np.asarray(paz.image.load(args.image))
masks = model(preprocess(image))
class_map = np.asarray(jp.argmax(jp.squeeze(masks, axis=0), axis=-1))
image = np.asarray(paz.image.resize_opencv(image, (H, W)))
paz.image.write(args.output, paz.draw.overlay_masks(image, class_map, colors))
