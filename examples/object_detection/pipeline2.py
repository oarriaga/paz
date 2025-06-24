import jax.numpy as jp
import numpy as np
import paz
import jax


def resize_and_pad(images, detections, H, W, pad_size):
    resized_images, resized_labels = [], []
    for sample in zip(images, detections):
        sample = [jp.array(x) for x in sample]
        sample_image, sample_label = paz.detection.resize(*sample, H, W)
        sample_label = jp.array(sample_label, dtype=jp.float32)
        sample_label = paz.detection.pad(sample_label, pad_size, "constant", -1)
        resized_images.append(sample_image)
        resized_labels.append(sample_label)
    return jp.array(resized_images), jp.array(resized_labels)


def add_background_class(detections):
    boxes, class_args = detections = paz.detection.split(detections)
    detections = paz.detection.merge(boxes, class_args + 1)
    return detections


def preprocess_detections(
    detections, prior_boxes, num_classes, IOU, variances, H, W
):
    detections = paz.detection.normalize(detections, H, W)
    detections = add_background_class(detections)
    detections = paz.detection.match(detections, prior_boxes, IOU)
    detections = paz.detection.encode(detections, prior_boxes, variances)
    # internally increase number of classes by 1 to account for background class
    detections = paz.detection.to_one_hot(jp.array(detections), num_classes + 1)
    return jp.array(detections)


def preprocess_image(key, image, mean, augment=True):
    if augment:
        image = paz.image.augment_color(key, image)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    return image


def preprocess_batch(
    key,
    images,
    detections,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
    augment=True,
):

    images = [paz.image.load(image) for image in images]
    images, detections = resize_and_pad(images, detections, H, W, max_num_boxes)

    preprocess_images = jax.jit(
        jax.vmap(paz.lock(preprocess_image, jp.array(mean), augment), (0, 0))
    )
    images = preprocess_images(jax.random.split(key, len(images)), images)
    args = prior_boxes, num_classes, match_IOU, variances, H, W
    detections = jax.jit(jax.vmap(paz.lock(preprocess_detections, *args)))(
        detections
    )
    return np.array(images, dtype="float32"), np.array(
        detections, dtype="float32"
    )
