# TODO verify that training pipeline is still valid with shapes dataset
# TODO make function that filters classes
# TODO map invalid to background mask and remove them from masks altogether
# TODO make functions that filters bounding boxes
# TODO debug detector loss
# TODO build callback for drawing segmentations masks
# TODO build callback for drawing internal masks
from collections import namedtuple
from functools import partial
import tensorflow as tf
import paz

from keras import ops
from keras_cv.layers import Resizing, RandomColorJitter
from keras_cv import visualization

import numpy as np


def color_map_to_class_arg(mask, colormap_to_class):
    class_masks = ops.zeros(ops.shape(mask)[:2])
    for color, class_arg in colormap_to_class.items():
        color = ops.cast(ops.convert_to_tensor(color), "uint8")
        is_class = ops.all(mask == color, axis=-1)
        class_masks = ops.where(is_class, class_arg, class_masks)
    return class_masks


def load_image(image_path, channels=3):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=channels)
    return image


def load_masks(masks_path, color_map):
    masks = tf.io.read_file(masks_path)
    masks = tf.image.decode_png(masks, channels=3, dtype="uint8")
    masks = color_map_to_class_arg(masks, color_map)
    return masks


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
    plt.show()


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


def Load(color_map):
    def apply(x):
        image_path = x["images"]
        masks_path = x["segmentation_masks"]
        image = load_image(image_path, 3)
        masks = load_masks(masks_path, color_map)[..., None]
        x["images"] = image
        x["segmentation_masks"] = masks
        return x

    return apply


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


def Pipeline(data, H, W, batch_size, color_to_class):
    data = data.map(wrap, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(Load(color_to_class), num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(Resize(H, W), num_parallel_calls=tf.data.AUTOTUNE)
    data = data.shuffle(batch_size * 4)
    data = data.ragged_batch(batch_size, drop_remainder=True)
    data = data.map(AugmentImage(), num_parallel_calls=tf.data.AUTOTUNE)
    return data


def plot_sample(sample, class_map):
    masks = ops.convert_to_numpy(sample["segmentation_masks"])[0]
    image = ops.convert_to_numpy(sample["images"])[0].astype("uint8")
    labels = (np.unique(masks).astype(int)).tolist()
    # labels.remove(21)
    # labels.remove(0)
    print(labels)
    figure, axes = plt.subplots(1, len(labels) + 1)
    axes[0].imshow(image)
    for axis, label in zip(axes[1:], labels):
        is_class = masks == label
        axis.imshow(np.where(is_class, 255, 0).astype(np.uint8))
        if label == 0:
            class_name = "background"
        elif label == 21:
            class_name = "invalid"
        else:
            class_name = f"{class_map[label - 1]}"
        axis.set_title(class_name)
    plt.show()


def build_valid_mask(class_args, valid_class_args):
    is_valid_mask = []
    for class_arg in class_args:
        if class_arg in valid_class_args:
            is_valid_mask.append(True)
        else:
            is_valid_mask.append(False)
    return is_valid_mask


def all_are_invalid(is_valid_mask):
    # TODO remove
    return not all(not is_valid for is_valid in is_valid_mask)


class MultiList(object):
    def __init__(self, num_lists):
        self.lists = [[] for _ in range(num_lists)]
        print(self.lists)

    def append(self, *args):
        for arg_list, arg in zip(self.lists, args):
            arg_list.append(arg)
            print("hi", self.lists)


def filter_sample(sample_class_args, sample_boxes, is_valid_mask):
    masked_detections = MultiList(2)
    iterator = zip(is_valid_mask, sample_class_args, sample_boxes)
    for is_valid, instance_class_arg, instance_box in iterator:
        if is_valid:
            masked_detections.append(instance_class_arg, instance_box)
    return masked_detections.lists


def filter_by_class(images, masks, boxes, class_args, valid_class_args):
    filtered_data = MultiList(4)
    iterator = zip(images, masks, class_args, boxes)
    for image, mask, sample_class_args, sample_boxes in iterator:
        is_valid_mask = build_valid_mask(class_args, valid_class_args)
        if any(is_valid_mask):
            filtered_sample_class_args, filtered_sample_boxes = filter_sample(
                sample_class_args, sample_boxes, is_valid_mask
            )
            filtered_data.append(
                image, mask, filtered_sample_class_args, filtered_sample_boxes
            )
    return filtered_data.lists


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    data = paz.datasets.load("VOC2007", "test", "segmentation")
    # images, masks, boxes, class_args = data
    images, masks, boxes, class_args = filter_by_class(*data, [0])

    # data = TFDataset(images, class_args, boxes, masks)
    # color_to_class = paz.datasets.voc.colormap_to_class()
    # data = Pipeline(data, 340, 340, 4, color_to_class)
    # class_map = paz.datasets.class_map("VOC2007")

    # plot_labels(data, "xyxy", class_map)
    # class_map = paz.datasets.class_map("VOC2007")

    # for sample in data:
    #     plot_sample(sample, class_map)
