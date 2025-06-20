import jax
import jax.numpy as jp
import paz


def compute_samples(positive_ratio, batch_size):
    num_positives = int(positive_ratio * batch_size)
    num_negatives = batch_size - num_positives
    return num_positives, num_negatives


def compute_num_positives(positive_ratio, batch_size):
    return int(positive_ratio * batch_size)


def compute_num_negatives(positive_ratio, batch_size):
    num_positives = compute_num_positives(positive_ratio, batch_size)
    num_negatives = batch_size - num_positives
    return num_negatives


def augment(key, image, angle_range=(-jp.pi / 8, jp.pi / 8)):
    key_0, key_1, key_2, key_3, key_4, key_5 = jax.random.split(key, 6)
    image = paz.image.random_flip_left_right(key_0, image)
    image = paz.image.random_saturation(key_1, image)
    image = paz.image.random_brightness(key_2, image)
    image = paz.image.random_contrast(key_3, image, 0.9, 1.1)
    image = paz.image.random_hue(key_4, image)
    # image = paz.image.random_rotation(key_5, image, *angle_range)
    return image


def augment_batch(key, images, angle_range=(-jp.pi / 8, jp.pi / 8)):
    keys = jax.random.split(key, len(images))
    images = jax.vmap(augment)(keys, images)
    return images


def shuffle(key, images, labels):
    shuffled_args = jax.random.permutation(key, jp.arange(len(images)))
    images = images[shuffled_args]
    labels = labels[shuffled_args]
    return images, labels


def label(positive_images, negative_images):
    positive_labels = jp.full(len(positive_images), 1.0)
    negative_labels = jp.full(len(negative_images), 0.0)
    images = jp.concatenate([negative_images, positive_images], axis=0)
    labels = jp.concatenate([negative_labels, positive_labels], axis=0)
    return images, labels


def boxes_to_images(key, boxes, image, box_size, pad, augment=True):
    boxes = paz.boxes.square(boxes)
    boxes = paz.boxes.set_size(boxes, *box_size)
    images = paz.boxes.crop_with_pad(boxes, image, *box_size, pad)
    if augment:
        images = augment_batch(key, images)
    return images


def batch(
    key,
    detections,
    image,
    augment=True,
    box_size=(128, 128),
    positive_ratio=0.5,
    batch_size=32,
    scale_range=(0.8, 1.4),
    shift_range=(-20, 20),
    pad=jp.array(paz.image.RGB_IMAGENET_MEAN, dtype="uint8"),
):
    keys = jax.random.split(key, 5)
    size = paz.image.get_size(image)
    positive_boxes = paz.detection.get_boxes(detections)
    positive_boxes = paz.boxes.denormalize(positive_boxes, *size)

    num_negatives = compute_num_negatives(positive_ratio, batch_size)
    args = (positive_boxes, *size, box_size, num_negatives, num_negatives * 3)
    negative_boxes = paz.boxes.sample_negatives(keys[0], *args)

    num_positives = compute_num_positives(positive_ratio, batch_size)
    args = (positive_boxes, *size, num_positives, scale_range, shift_range)
    positive_boxes = paz.boxes.sample_positives(keys[1], *args)

    to_images = paz.lock(boxes_to_images, image, box_size, pad, augment)
    positive_images = to_images(keys[2], positive_boxes)
    negative_images = to_images(keys[3], negative_boxes)
    images, labels = label(positive_images, negative_images)
    images, labels = shuffle(keys[4], images, labels)
    return images, labels
