import cv2
import jax.numpy as jp
import numpy as np
import paz
import jax


def preprocess_batch(key, images, detections, H, W, prior_boxes, num_classes,
                     match_IOU, variances, mean, max_num_boxes, augment=True):
    pipeline = build_pipeline(H, W, prior_boxes, num_classes, match_IOU,
                              variances, mean)
    images, detections = load_batch(images, detections, H, W, max_num_boxes)
    images = jp.asarray(images, jp.float32)
    detections = jp.asarray(detections, jp.float32)
    if augment:
        keys = jax.random.split(key, len(images))
        images, detections = pipeline["augment"](keys, images, detections)
    else:
        images, detections = pipeline["preprocess"](images, detections)
    return np.asarray(images, "float32"), np.asarray(detections, "float32")


def load_batch(paths, detections, H, W, max_boxes):
    """CPU I/O boundary: decode and resize each image with cv2 (releases the
    GIL, no host<->device bounce), scale boxes to HxW pixels and pad to a fixed
    count. Everything downstream is JAX on device."""
    images, labels = [], []
    for path, detection in zip(paths, detections):
        image = cv2.imread(str(path))
        H_now, W_now = image.shape[:2]
        images.append(cv2.resize(image, (W, H))[:, :, ::-1])  # BGR -> RGB
        detection = np.asarray(detection, "float32")
        scale = np.array([W / W_now, H / H_now, W / W_now, H / H_now])
        boxes = detection[:, :4] * scale
        label = np.concatenate([boxes, detection[:, 4:]], axis=1)[:max_boxes]
        padding = ((0, max_boxes - len(label)), (0, 0))
        labels.append(np.pad(label, padding, constant_values=-1))
    return np.ascontiguousarray(np.stack(images)), np.stack(labels)


# The whole batch preprocess + augmentation is a single jit(vmap(...)) built
# once per configuration, so XLA compiles it once and it runs on device.
_PIPELINE = {}


def build_pipeline(H, W, prior_boxes, num_classes, IOU, variances, mean):
    key = (H, W, num_classes, IOU, tuple(variances))
    if key not in _PIPELINE:
        mean = jp.asarray(mean, jp.float32)
        args = prior_boxes, num_classes, IOU, variances, mean, H, W
        augment = jax.jit(jax.vmap(paz.lock(augment_sample, *args), (0, 0, 0)))
        preprocess = jax.jit(jax.vmap(paz.lock(preprocess_sample, *args)))
        _PIPELINE[key] = {"augment": augment, "preprocess": preprocess}
    return _PIPELINE[key]


def augment_sample(key, image, detection, prior_boxes, num_classes, IOU,
                   variances, mean, H, W):
    detection = paz.detection.normalize(detection, H, W)
    image, detection = paz.detection.augment_detection(
        key, image, detection, mean)
    image = to_model_input(image, mean)
    detection = encode_detection(detection, prior_boxes, num_classes, IOU,
                                 variances)
    return image, detection


def preprocess_sample(image, detection, prior_boxes, num_classes, IOU,
                      variances, mean, H, W):
    detection = paz.detection.normalize(detection, H, W)
    image = to_model_input(image, mean)
    detection = encode_detection(detection, prior_boxes, num_classes, IOU,
                                 variances)
    return image, detection


def to_model_input(image, mean):
    return paz.image.subtract_mean(paz.image.RGB_to_BGR(image), mean)


def encode_detection(detection, prior_boxes, num_classes, IOU, variances):
    detection = add_background_class(detection)
    detection = paz.detection.match(detection, prior_boxes, IOU)
    detection = paz.detection.encode(detection, prior_boxes, variances)
    return paz.detection.to_one_hot(detection, num_classes + 1)


def add_background_class(detections):
    boxes, class_args = paz.detection.split(detections)
    return paz.detection.merge(boxes, class_args + 1)
