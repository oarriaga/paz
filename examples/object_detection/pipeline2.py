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


def preprocess_detections(detections, prior_boxes, num_classes, IOU, variances):
    detections = add_background_class(detections)
    detections = paz.detection.match(detections, prior_boxes, IOU)
    detections = paz.detection.encode(detections, prior_boxes, variances)
    # internally increase number of classes by 1 to account for background class
    detections = paz.detection.to_one_hot(jp.array(detections), num_classes + 1)
    return jp.array(detections)


def preprocess_image(image, mean):
    image = paz.cast(image, jp.float32)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    return image


# Jitted batch stages are built once per configuration and reused across
# batches, so XLA compiles them a single time (cache_size stays at 1).
_PIPELINE = {}


def build_pipeline(prior_boxes, num_classes, IOU, variances, mean):
    key = (num_classes, IOU, tuple(variances))
    if key not in _PIPELINE:
        mean = jp.asarray(mean, jp.float32)
        detection_args = prior_boxes, num_classes, IOU, variances
        _PIPELINE[key] = {
            "normalize": jax.jit(jax.vmap(paz.detection.normalize, (0, None,
                                                                     None))),
            "augment": jax.jit(jax.vmap(
                paz.detection.augment_detection, (0, 0, 0, None))),
            "image": jax.jit(jax.vmap(paz.lock(preprocess_image, mean))),
            "detection": jax.jit(jax.vmap(
                paz.lock(preprocess_detections, *detection_args))),
            "mean": mean,
        }
    return _PIPELINE[key]


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
    pipeline = build_pipeline(prior_boxes, num_classes, match_IOU, variances,
                              mean)
    images = [paz.image.load(image) for image in images]
    images, detections = resize_and_pad(images, detections, H, W, max_num_boxes)
    images = images.astype(jp.float32)
    detections = pipeline["normalize"](detections, H, W)
    if augment:
        keys = jax.random.split(key, len(images))
        images, detections = pipeline["augment"](keys, images, detections,
                                                  pipeline["mean"])
    images = pipeline["image"](images)
    detections = pipeline["detection"](detections)
    return np.array(images, "float32"), np.array(detections, "float32")
