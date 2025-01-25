# most likely the class maps are wrong. Verify them using
# https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae

import cv2
import tensorflow as tf
import paz

from keras import ops
from keras_cv.layers import Resizing, RandomColorJitter
from keras_cv import visualization

import numpy as np
from PIL import Image


def remove_masks_colormap(filepath):
    # https://stackoverflow.com/questions/51702670/tensorflow-deeplab-image-colormap-removal-confusion
    image = np.array(Image.open(filepath))
    image = Image.fromarray(image.astype(dtype=np.uint8))
    image.save(filepath)


def _color_to_class(mask, colormap_to_class):
    class_masks = np.ones(mask.shape[:2]) * 21
    for color, class_arg in colormap_to_class.items():
        color = np.array(color).astype("uint8")
        is_class = np.all(mask == color, axis=-1)
        class_masks = np.where(is_class, class_arg, class_masks)
    return class_masks[..., 0:1]


def color_map_to_class_arg(mask, colormap_to_class):
    class_masks = ops.zeros(ops.shape(mask)[:2])
    for color, class_arg in colormap_to_class.items():
        color = ops.cast(ops.convert_to_tensor(color), "uint8")
        is_class = ops.all(mask == color, axis=-1)
        class_masks = ops.where(is_class, class_arg, class_masks)
    return class_masks


def plot_labels(dataset, bounding_box_format, class_map):
    sample = next(iter(dataset.take(1)))
    visualization.plot_bounding_box_gallery(
        images=sample["images"],
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=sample["bounding_boxes"],
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_map,
    )


def load_image(image_path, channels=3):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=channels)
    return image


def load_masks(masks_path, color_map):
    masks = tf.io.read_file(masks_path)
    masks = tf.image.decode_png(masks, channels=3, dtype="uint8")
    masks = color_map_to_class_arg(masks, color_map)
    return masks


def Resize(H, W):
    kwargs = {"pad_to_aspect_ratio": True, "bounding_box_format": "xyxy"}
    resize = Resizing(H, W, **kwargs)
    return resize


def AugmentImage():
    return RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=(-0.2, 0.2),
        contrast_factor=(0.5, 0.5),
        saturation_factor=(0.5, 0.9),
        hue_factor=(0.0, 0.0),
    )


def load(x, color_map):
    x["images"] = load_image(x["images"], 3)
    x["segmentation_masks"] = load_masks(x["segmentation_masks"], color_map)
    return x


def wrap(images, classes, boxes, masks):
    boxes = {"classes": classes, "boxes": boxes}
    return {
        "images": images,
        "bounding_boxes": boxes,
        "segmentation_masks": masks,
    }


def to_ragged(values):
    return tuple([tf.ragged.constant(value) for value in values])


def TFDataset(*args):
    return tf.data.Dataset.from_tensor_slices(to_ragged(args))


def Pipeline(data, H, W, batch_size):
    data = data.map(wrap, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(
        paz.lock(load, paz.datasets.voc.colormap_to_class()),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    data = data.shuffle(batch_size * 4)
    data = data.ragged_batch(batch_size, drop_remainder=True)
    # data = data.map(AugmentImage(), num_parallel_calls=tf.data.AUTOTUNE)
    # data = data.map(Resize(H, W), num_parallel_calls=tf.data.AUTOTUNE)
    return data


def plot_masks(image, masks, class_map):
    labels = (np.unique(masks).astype(int)).tolist()
    labels.remove(21)
    labels.remove(0)
    print(labels)
    figure, axes = plt.subplots(1, len(labels) + 1)
    axes[0].imshow(image)
    for axis, label in zip(axes[1:], labels):
        # class_name = class_map[label]
        # axis.imshow(np.where(masks == label, 255, 0).astype(np.uint8))
        # is_class = np.all(masks == label, axis=-1)
        is_class = masks == label
        axis.imshow(np.where(is_class, 255, 0).astype(np.uint8))
        axis.set_title(f"{class_map[label - 1]}")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

data = paz.datasets.load("VOC2007", "trainval", "segmentation")
images, masks, boxes, class_args = data
data = TFDataset(images, class_args, boxes, masks)
data = Pipeline(data, 340, 340, 4)
class_map = paz.datasets.class_map("VOC2007")
plot_labels(data, "xyxy", class_map)
plt.show()

# m = load_masks(masks[0], paz.datasets.voc.colormap_to_class())
# i = load_image(images[0])
# plot_masks(i, m, paz.datasets.class_map("VOC2007"))
for sample in data:
    m = ops.convert_to_numpy(sample["segmentation_masks"])[0]
    i = ops.convert_to_numpy(sample["images"])[0]
    print("Masks shape", m.shape)
    print("Image shape", i.shape)
    plot_masks(i, m, paz.datasets.class_map("VOC2007"))
# train_pipeline = data.map(wrap, num_parallel_calls=tf.data.AUTOTUNE)
# boxes = paz.boxes.pad_data(boxes, 32)
# class_args = paz.classes.pad_data(class_args, 32)
# data = tf.data.Dataset.from_tensor_slices((images, masks, boxes, class_args))

# m = ops.convert_to_numpy(m)
# color_to_class = paz.datasets.voc.colormap_to_class()
# m_i = _to_int_mask(m, color_to_class)
