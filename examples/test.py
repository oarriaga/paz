from keras import ops, Model
import matplotlib.pyplot as plt
import numpy as np
from keras_cv.src.visualization.draw_bounding_boxes import draw_bounding_boxes

from functools import partial
import tensorflow as tf
import paz

from keras import ops
from keras_cv.models import YOLOV8Backbone, YOLOV8Segmentation
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


def make_mosaic(images, shape, border=0):
    num_images, H, W, num_channels = images.shape
    num_rows, num_cols = shape
    if num_images > (num_rows * num_cols):
        raise ValueError("Number of images is bigger than shape")

    total_rows = (num_rows * H) + ((num_rows - 1) * border)
    total_cols = (num_cols * W) + ((num_cols - 1) * border)
    mosaic = np.ones((total_rows, total_cols, num_channels))

    padded_H = H + border
    padded_W = W + border

    for image_arg, image in enumerate(images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        mosaic[
            row * padded_H : row * padded_H + H,
            col * padded_W : col * padded_W + W,
            :,
        ] = image
    return mosaic


def visualize_prototypes(prototypes, H_mosaic=8, W_mosaic=4):
    mosaics = []
    if prototypes.shape[0] is None:
        return
    prototypes = ops.convert_to_numpy(prototypes)
    batch, H, W, num_masks = prototypes.shape
    prototypes = np.moveaxis(prototypes, 3, 1)
    prototypes = np.expand_dims(prototypes, -1)
    for batch_arg in range(prototypes.shape[0]):
        mosaic = make_mosaic(prototypes[batch_arg], (H_mosaic, W_mosaic), 2)
        mosaics.append(mosaic)
    mosaics = np.array(mosaics)
    mosaics = make_mosaic(mosaics, (1, len(mosaics)), 4)
    plt.figure(figsize=(10, 10))
    plt.imshow(mosaics)
    axis = plt.gca()
    axis.set_xticks([])
    axis.set_yticks([])
    plt.show()


def resize_masks(masks, H, W):
    masks = ops.image.resize(masks, (H, W))
    masks = ops.convert_to_numpy(masks)
    return masks


def postprocess_masks(masks, scores, H, W):
    is_valid_mask = scores > -1
    masks = masks[is_valid_mask][..., None]
    masks = resize_masks(masks, H, W)
    return masks


def postprocess_masks_batch(batch_masks, batch_scores, H, W):
    preprocessed_batch_masks = []
    for masks, scores in zip(batch_masks, batch_scores):
        masks = postprocess_masks(masks, scores, H, W)
        masks = np.squeeze(masks, axis=-1)
        preprocessed_batch_masks.append(masks)
    return preprocessed_batch_masks


def build_masks(image, masks, H, W, alpha=0.35, mask_treshold=0.5):
    image_masks = np.ones((H, W, 4))
    image_masks[:, :, 3] = 0
    for mask in masks:
        color_mask = np.array([0.1, 0.9, 0.1, alpha])
        mask = mask > mask_treshold
        image_masks[mask] = color_mask
    return image_masks


H, W = 510, 510
data = paz.datasets.load("VOC2007", "test", "segmentation")
images, masks, boxes, class_args = data
data = TFDataset(images, class_args, boxes, masks)
color_to_class = paz.datasets.voc.colormap_to_class()
test_data = Pipeline(data, H, W, 4, color_to_class)
class_map = paz.datasets.class_map("VOC2007")

input_shape = (H, W, 3)
backbone = YOLOV8Backbone.from_preset(
    "yolo_v8_m_backbone_coco", input_shape=input_shape, load_weights=True
)
model = YOLOV8Segmentation(
    num_classes=len(class_map),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=2,
    prototype_dimension=32,
    trainable_backbone=True,
)

model.load_weights("YOLOV8-SEG-VOC.weights.h5")


inputs = next(iter(test_data.take(1)))
get_prototypes = Model(
    inputs=model.input, outputs=model.get_layer("prototypes").output
)
prototypes = get_prototypes(inputs["images"])

visualize_prototypes(prototypes)

y_pred = model.predict(inputs["images"])
images = ops.convert_to_numpy(inputs["images"])
batch_masks = y_pred["masks"]
batch_score = y_pred["confidence"]
masks = postprocess_masks_batch(batch_masks, batch_score, H, W)

images_with_boxes = draw_bounding_boxes(
    images,
    y_pred,
    (0, 255, 0),
    "xyxy",
    class_mapping=class_map,
    font_scale=0.40,
    text_thickness=1,
)

batch_size = len(images_with_boxes)
mosaic = []
for batch_arg in range(batch_size):
    image = images_with_boxes[batch_arg]
    mask = masks[batch_arg]
    mask = build_masks(image, mask, H, W, 0.75, 0.6)
    alpha = mask[:, :, 3]
    combined_image = (1 - alpha[:, :, np.newaxis]) * (image / 255) + alpha[
        :, :, np.newaxis
    ] * mask[:, :, :3]
    mosaic.append(images[batch_arg] / 255)
    mosaic.append(image / 255)
    mosaic.append(combined_image)

mosaic = make_mosaic(np.array(mosaic), (batch_size, 3), 5)
plt.figure(figsize=(10, 10))
plt.imshow(mosaic)
