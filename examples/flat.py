import os
from functools import partial
from xml.etree import ElementTree
from functools import partial

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras import ops
from keras.utils import get_file
from keras.optimizers import Adam
from keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from keras_cv import visualization
from keras_cv.layers import Resizing, RandomColorJitter
from keras_cv.models import YOLOV8Backbone, YOLOV8Segmentation


def build_name_to_arg(class_names):
    return dict(zip(class_names, range(len(class_names))))


def build_arg_to_name(class_names):
    return dict(zip(range(len(class_names)), class_names))


def get_class_names():
    return [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]


def get_class_map():
    class_names = get_class_names()
    return build_arg_to_name(class_names)


def parse_box(box):
    # matlab start indices at 1 thus we need to subtract 1
    x_min = int(box.find("xmin").text) - 1
    y_min = int(box.find("ymin").text) - 1
    x_max = int(box.find("xmax").text) - 1
    y_max = int(box.find("ymax").text) - 1
    return [x_min, y_min, x_max, y_max]


def parse_XML(name_to_arg, XML_filename):
    tree = ElementTree.parse(XML_filename)
    image_name = tree.find("filename").text
    boxes, class_args = [], []
    for detection in tree.findall("object"):
        class_arg = name_to_arg[detection.find("name").text]
        difficult = int(detection.find("difficult").text)
        box = parse_box(detection.find("bndbox"))
        if not difficult:
            boxes.append(box)
            class_args.append(class_arg)
    return image_name, boxes, class_args


def validate_inputs(name, split, task):
    assert name in ["VOC2007", "VOC2012"]
    assert split in ["trainval", "test"]
    assert task in ["detection", "segmentation"]


def parse_line(path, name, line):
    return os.path.join(path, name, "Annotations", line.strip() + ".xml")


def build_label_path(path, name, split, task):
    task = "Segmentation" if task == "segmentation" else "Main"
    return os.path.join(path, name, f"ImageSets/{task}/{split}.txt")


def get_label_paths(path, name, split, task):
    split_filename = build_label_path(path, name, split, task)
    return [parse_line(path, name, line) for line in open(split_filename)]


def get_URL(name, split):
    base_URL = "http://host.robots.ox.ac.uk/pascal/VOC/"
    if (name == "VOC2007") and (split == "trainval"):
        URL = base_URL + "voc2007/VOCtrainval_06-Nov-2007.tar"
    elif (name == "VOC2012") and (split == "trainval"):
        URL = base_URL + "voc2012/VOCtrainval_11-May-2012.tar"
    elif (name == "VOC2007") and (split == "test"):
        URL = base_URL + "voc2007/VOCtest_06-Nov-2007.tar"
    else:
        raise ValueError("Invalid split")
    return URL


def download(name, split):
    origin = get_URL(name, split)
    filepath = get_file(origin=origin, extract=True)
    filepath = os.path.join(os.path.dirname(filepath), "VOCdevkit")
    return filepath


def strip_extension(file_path):
    base_name, _ = os.path.splitext(file_path)
    return base_name


def load(name, split="trainval", task="detection"):
    """Loads VOC2007 or VOC2012 with the trainval or test split.

    # Arguments
        name: String. Either `VOC2007` or `VOC2012`.
        split: String. Either `trainval` or `test`.
        task: String. Either `detection` or `segmentation`.

    # Returns
        Lists of image paths, boxes in xyxy format, and classes arguments.
    """
    validate_inputs(name, split, task)
    path = download(name, split)
    image_root = os.path.join(path, name, "JPEGImages")
    masks_root = os.path.join(path, name, "SegmentationClass")
    class_names = get_class_names()
    parse = partial(parse_XML, build_name_to_arg(class_names))
    image_paths, masks_paths, boxes, class_args = [], [], [], []
    for label_path in get_label_paths(path, name, split, task):
        image_name, image_boxes, image_class_args = parse(label_path)
        if len(image_boxes) != 0:
            image_name = strip_extension(image_name)
            image_paths.append(os.path.join(image_root, image_name + ".jpg"))
            masks_paths.append(os.path.join(masks_root, image_name + ".png"))
            boxes.append(image_boxes)
            class_args.append(image_class_args)
        else:
            print(f"Image {image_name} had not boxes.")
    return image_paths, class_args, boxes, masks_paths


def get_colormap_to_class():
    return {
        (0, 0, 0): 0,
        (128, 0, 0): 1,
        (0, 128, 0): 2,
        (128, 128, 0): 3,
        (0, 0, 128): 4,
        (128, 0, 128): 5,
        (0, 128, 128): 6,
        (128, 128, 128): 7,
        (64, 0, 0): 8,
        (192, 0, 0): 9,
        (64, 128, 0): 10,
        (192, 128, 0): 11,
        (64, 0, 128): 12,
        (192, 0, 128): 13,
        (64, 128, 128): 14,
        (192, 128, 128): 15,
        (0, 64, 0): 16,
        (128, 64, 0): 17,
        (0, 192, 0): 18,
        (128, 192, 0): 19,
        (0, 64, 128): 20,
        (224, 224, 192): 21,
    }


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
    labels.remove(21)
    labels.remove(0)
    print(labels)
    figure, axes = plt.subplots(1, len(labels) + 1)
    axes[0].imshow(image)
    for axis, label in zip(axes[1:], labels):
        is_class = masks == label
        axis.imshow(np.where(is_class, 255, 0).astype(np.uint8))
        axis.set_title(f"{class_map[label - 1]}")
    plt.show()


prototype_dimension = 64
batch_size = 4
FPN_depth = 1
backbone_name = "yolo_v8_s_backbone_coco"
trainable_backbone = False

epochs = 50
learning_rate = 1e-3
global_clipnorm = 10.0
stop_patience = 5
stop_delta = 1e-3
reduce_LR_factor = 0.1
reduce_LR_patience = 3

run_eagerly = False
jit_compile = False
H, W, _ = input_shape = (340, 340, 3)
masks_loss = "binary_crossentropy"
score_loss = "binary_crossentropy"
boxes_loss = "ciou"

class_map = get_class_map()
color_map = get_colormap_to_class()


train_data_07 = load("VOC2007", "trainval", "segmentation")
train_data_07 = TFDataset(*train_data_07)
train_data_07 = Pipeline(train_data_07, H, W, batch_size, color_map)


train_data_12 = load("VOC2007", "trainval", "segmentation")
train_data_12 = TFDataset(*train_data_12)
train_data_12 = Pipeline(train_data_12, H, W, batch_size, color_map)

train_data = train_data_07.concatenate(train_data_12)
plot_labels(train_data, "xyxy", class_map)

test_data = load("VOC2007", "test", "segmentation")
test_data = TFDataset(*test_data)
test_data = Pipeline(test_data, H, W, batch_size, color_map)
plot_labels(test_data, "xyxy", class_map)


backbone = YOLOV8Backbone.from_preset(backbone_name, input_shape=input_shape)
model = YOLOV8Segmentation(
    num_classes=len(get_class_names()),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=FPN_depth,
    prototype_dimension=prototype_dimension,
    trainable_backbone=trainable_backbone,
)

optimizer = Adam(learning_rate, global_clipnorm=global_clipnorm)
model.compile(
    boxes_loss,
    score_loss,
    masks_loss,
    optimizer=optimizer,
    run_eagerly=run_eagerly,
    jit_compile=jit_compile,
)
model.summary()


log = CSVLogger("optimization.csv")
checkpoint = ModelCheckpoint(
    "model.weights.h5", verbose=1, save_best_only=True, save_weights_only=True
)
stop = EarlyStopping("val_loss", stop_delta, stop_patience, 1)
reduce_LR = ReduceLROnPlateau(
    "val_loss", reduce_LR_factor, reduce_LR_patience, 1
)

model.fit(
    train_data,
    epochs=epochs,
    callbacks=[log, checkpoint, reduce_LR, stop],
    validation_data=test_data,
)
