import jax
import jax.numpy as jp
import numpy as np
import paz


def preprocess_batch(key, images, detections, H, W, prior_boxes, num_classes,
                     match_IOU, variances, mean, max_num_boxes, augment=True):
    images, detections = load_batch(images, detections, H, W, max_num_boxes)
    images = jp.asarray(images, jp.float32)
    detections = jp.asarray(detections, jp.float32)
    keys = jax.random.split(key, len(images))
    mean = jp.asarray(mean, jp.float32)
    images, detections = transform_batch(
        keys, images, detections, prior_boxes, mean, num_classes, match_IOU,
        tuple(variances), augment)
    return np.asarray(images, "float32"), np.asarray(detections, "float32")


def load_batch(paths, detections, H, W, max_boxes):
    """CPU I/O boundary: read and resize each image, scale boxes to H x W
    pixels and pad to a fixed count. Everything downstream is JAX on device."""
    images, labels = [], []
    for path, detection in zip(paths, detections):
        image = paz.image.load(path)
        H_now, W_now = paz.image.get_size(image)
        images.append(np.asarray(paz.image.resize(image, (H, W))))
        detection = np.asarray(detection, "float32")
        scale = np.array([W / W_now, H / H_now, W / W_now, H / H_now])
        boxes = detection[:, :4] * scale
        label = np.concatenate([boxes, detection[:, 4:]], axis=1)[:max_boxes]
        padding = ((0, max_boxes - len(label)), (0, 0))
        labels.append(np.pad(label, padding, constant_values=-1))
    return np.stack(images), np.stack(labels)


# One jitted vmap over the batch. JAX compiles and caches a variant per static
# configuration, so no hand-managed pipeline cache is needed.
@paz.partial(jax.jit, static_argnames=("num_classes", "match_IOU",
                                       "variances", "augment"))
def transform_batch(keys, images, detections, prior_boxes, mean, num_classes,
                    match_IOU, variances, augment):
    def transform(key, image, detection):
        H, W = paz.image.get_size(image)
        detection = paz.detection.normalize(detection, H, W)
        if augment:
            image, detection = paz.detection.augment_detection(
                key, image, detection, mean)
        image = preprocess_image(image, mean)
        detection = paz.detection.encode_detection(
            detection, prior_boxes, num_classes, match_IOU, variances)
        return image, detection

    return jax.vmap(transform)(keys, images, detections)


def preprocess_image(image, mean):
    return paz.image.subtract_mean(paz.image.RGB_to_BGR(image), mean)
