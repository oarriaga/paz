from tqdm import tqdm
import math
import jax
from deepfish import load
import jax.numpy as jp
import matplotlib.pyplot as plt
import paz
import plotter

key = jax.random.PRNGKey(777)
train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
images = train_images + valid_images
labels = train_labels + valid_labels

boxes_per_image = jp.array([len(label) for label in labels])


def display_images_with_labels(images, labels):
    num_images = images.shape[0]
    cols = min(int(math.ceil(math.sqrt(num_images))), 8)
    rows = int(math.ceil(num_images / cols))

    figure, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for image_arg in range(num_images):
        axes[image_arg].imshow(images[image_arg])
        axes[image_arg].set_title(f"{labels[image_arg]}")
        axes[image_arg].axis("off")

    for image_arg in range(num_images, len(axes)):
        axes[image_arg].axis("off")

    plt.tight_layout()
    return figure


def augment(key, image, angle_range=(-jp.pi / 8, jp.pi / 8)):
    key_0, key_1, key_2, key_3, key_4, key_5 = jax.random.split(key, 6)
    image = paz.image.random_flip_left_right(key_0, image)
    image = paz.image.random_saturation(key_1, image)
    image = paz.image.random_brightness(key_2, image)
    image = paz.image.random_contrast(key_3, image, 0.9, 1.1)
    image = paz.image.random_hue(key_4, image)
    # image = paz.image.random_rotation(key_5, image, *angle_range)
    return image


# validates that all images are the same size
# print(jp.unique(jp.array([paz.image.get_size(image)[0] for image in images])))

plotter.histogram_uniques(boxes_per_image, "Number of boxes")
plt.show()

# dataset_H, dataset_W = [], []
# for detections in tqdm(labels):
#     boxes = paz.boxes.denormalize(
#         paz.detection.get_boxes(detections), 1080, 1920
#     )
#     box_H, box_W = paz.boxes.compute_sizes(boxes, keepdims=False)
#     box_H = box_H.tolist()
#     box_W = box_W.tolist()
#     dataset_H = dataset_H + box_H
#     dataset_W = dataset_W + box_W
# dataset_H = jp.array(dataset_H)
# dataset_W = jp.array(dataset_W)

# plotter.histogram(dataset_H, "Boxes Height")
# plt.show()

# plotter.histogram(dataset_W, "Boxes Width")
# plt.show()


sample_arg = 0
box_H, box_W = 128, 128
detections = labels[sample_arg]
image = paz.image.load(images[sample_arg])
size = paz.image.get_size(image)
num_positive_samples = 100
positive_boxes = paz.detection.get_boxes(detections)
positive_boxes = paz.boxes.denormalize(positive_boxes, *size)

image_with_boxes = paz.draw.boxes(image, positive_boxes)
paz.image.show(image_with_boxes)

negative_boxes = paz.boxes.sample_negatives(
    key, positive_boxes, *size, (box_H, box_W), 200, 300
)

image_with_boxes = paz.draw.boxes(
    image_with_boxes, negative_boxes, color=(255, 0, 0)
)
paz.image.show(image_with_boxes)

box_H, box_W = paz.image.get_size(image)

positive_boxes = paz.boxes.sample_positives(
    key,
    positive_boxes,
    box_H,
    box_W,
    num_positive_samples,
    (0.9, 1.4),
    (0, 0),
)
image_with_boxes = paz.draw.boxes(image, positive_boxes)
image_with_boxes = paz.draw.boxes(
    image_with_boxes, negative_boxes, color=paz.draw.RED
)
paz.image.show(image_with_boxes)

# address edge case where all boxes are outside the image
positive_boxes = paz.boxes.square(positive_boxes)
positive_boxes = paz.boxes.filter_in_image(positive_boxes, box_H, box_W)
image_with_boxes = paz.draw.boxes(image, positive_boxes)

negative_boxes = paz.boxes.square(negative_boxes)
negative_boxes = paz.boxes.filter_in_image(negative_boxes, box_H, box_W)
image_with_boxes = paz.draw.boxes(
    image_with_boxes, negative_boxes, color=paz.draw.RED
)
paz.image.show(image_with_boxes)

mean = jp.array(paz.image.RGB_IMAGENET_MEAN, dtype="uint8")
positive_images = paz.boxes.crop_with_pad(positive_boxes, image, 128, 128, mean)
paz.image.show(paz.draw.mosaic(positive_images, border=10).astype("uint8"))
positive_images = jax.vmap(augment)(
    jax.random.split(key, len(positive_images)), positive_images
)
paz.image.show(paz.draw.mosaic(positive_images, border=10).astype(("uint8")))


negative_images = paz.boxes.crop_with_pad(negative_boxes, image, 128, 128, mean)
paz.image.show(
    paz.image.resize(
        paz.draw.mosaic(negative_images, border=10).astype("uint8"),
        (1080, 1080),
    ).astype("uint8")
)


negative_images = jax.vmap(augment)(
    jax.random.split(key, len(negative_images)), negative_images
)
paz.image.show(
    paz.image.resize(
        paz.draw.mosaic(negative_images, border=10).astype("uint8"),
        (1080, 1080),
    ).astype("uint8")
)


def compute_samples(positive_ratio, batch_size):
    num_positives = int(positive_ratio * batch_size)
    num_negatives = batch_size - num_positives
    return num_positives, num_negatives


def build_labels(key, positive_images, negative_images):
    positive_labels = jp.full(len(positive_images), 1.0)
    negative_labels = jp.full(len(negative_images), 0.0)
    images = jp.concatenate([negative_images, positive_images], axis=0)
    labels = jp.concatenate([negative_labels, positive_labels], axis=0)
    shuffled_args = jax.random.permutation(key, jp.arange(len(images)))
    images = images[shuffled_args]
    labels = labels[shuffled_args]
    return images, labels


def batch(
    key,
    detections,
    image,
    box_size=(256, 256),
    positive_ratio=0.5,
    batch_size=32,
    scale_range=(0.9, 1.4),
    shift_range=(0, 0),
    pad=jp.array(paz.image.RGB_IMAGENET_MEAN, dtype="uint8"),
):
    keys = jax.random.split(key, 5)
    size = paz.image.get_size(image)
    positive_boxes = paz.detection.get_boxes(detections)
    positive_boxes = paz.boxes.denormalize(positive_boxes, *size)
    num_positives, num_negatives = compute_samples(positive_ratio, batch_size)
    negative_boxes = paz.boxes.sample_negatives(
        keys[0],
        positive_boxes,
        *size,
        box_size,
        num_negatives,
        num_negatives * 3,
    )

    positive_boxes = paz.boxes.sample_positives(
        keys[1],
        positive_boxes,
        *size,
        num_positives,
        scale_range,
        shift_range,
    )

    positive_boxes = paz.boxes.square(positive_boxes)
    positive_boxes = paz.boxes.set_size(positive_boxes, *box_size)
    negative_boxes = paz.boxes.square(negative_boxes)
    negative_boxes = paz.boxes.set_size(negative_boxes, *box_size)
    positive_images = paz.boxes.crop_with_pad(
        positive_boxes, image, *box_size, pad
    )
    positive_images = jax.vmap(augment)(
        jax.random.split(keys[2], len(positive_images)), positive_images
    )

    negative_images = paz.boxes.crop_with_pad(
        negative_boxes, image, *box_size, pad
    )
    negative_images = jax.vmap(augment)(
        jax.random.split(keys[3], len(negative_images)), negative_images
    )
    return build_labels(keys[4], positive_images, negative_images)


sample_arg = 88
batch_images, batch_labels = jax.jit(batch)(
    key, labels[sample_arg], paz.image.load(images[sample_arg])
)
paz.image.show(paz.draw.mosaic(batch_images.astype("uint8")).astype("uint8"))
display_images_with_labels(batch_images, batch_labels)
plt.show()
