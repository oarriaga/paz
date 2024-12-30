import tensorflow as tf
import paz

from keras_cv.layers import Resizing, RandomColorJitter
from keras_cv import visualization


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
        hue_factor=(0.0, 0.0))


def to_dictionary(images, classes, boxes, masks):
    images = load_image(images, 3)
    masks = load_image(masks, 1)
    boxes = {"classes": classes, "boxes": boxes}
    return {"images": images,
            "bounding_boxes": boxes,
            "segmentation_masks": masks}


def to_ragged(values):
    return tuple([tf.ragged.constant(value) for value in values])


def TFDataset(*args):
    return tf.data.Dataset.from_tensor_slices(to_ragged(args))


def preprocess(data, H, W, batch_size):
    data = data.map(to_dictionary, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.shuffle(batch_size * 4)
    data = data.ragged_batch(batch_size, drop_remainder=True)
    data = data.map(AugmentImage(), num_parallel_calls=tf.data.AUTOTUNE)
    # data = data.map(Resize(H, W), num_parallel_calls=tf.data.AUTOTUNE)
    return data


import matplotlib.pyplot as plt
data = paz.datasets.load("VOC2007", "trainval", "segmentation")
images, masks, boxes, class_args = data
data = TFDataset(images, class_args, boxes, masks)
data = preprocess(data, 340, 340, 4)
class_map = paz.datasets.class_map("VOC2007")
plot_labels(data, "xyxy", class_map)
plt.show()
# x = next(iter(data.take(1)))
# train_pipeline = data.map(wrap, num_parallel_calls=tf.data.AUTOTUNE)
# boxes = paz.boxes.pad_data(boxes, 32)
# class_args = paz.classes.pad_data(class_args, 32)
# data = tf.data.Dataset.from_tensor_slices((images, masks, boxes, class_args))
