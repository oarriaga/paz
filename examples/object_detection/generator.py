import math
from keras.utils import PyDataset
import jax.numpy as jp
import jax
import paz


def pad(boxes, class_args, pad_size, pad_value):

    def pad_sample(boxes, class_args):
        boxes = jp.array(boxes)
        class_args = jp.array(class_args).reshape(-1, 1)
        detections = paz.detection.merge(boxes, class_args)
        detections = paz.detection.pad(detections, pad_size, "edge")
        return detections

    return jp.array([pad_sample(*sample) for sample in zip(boxes, class_args)])


def add_background_class(detections):
    boxes, class_args = detections = paz.detection.split(detections)
    detections = paz.detection.merge(boxes, class_args + 1)
    return detections


def preprocess_detections(detections, prior_boxes, num_classes, IOU, variances):
    detections = paz.detection.normalize(detections, 300, 300)
    detections = add_background_class(detections)
    detections = paz.detection.match(detections, prior_boxes, IOU)
    detections = paz.detection.encode(detections, prior_boxes, variances)
    # internally increase number of classes by 1 to account for background class
    detections = paz.detection.to_one_hot(detections, num_classes + 1)
    return detections


def preprocess_image(key, image, mean):
    image = paz.image.augment_color(key, image)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    return image


def resize_boxes(image, boxes, H, W):
    boxes = jp.array(boxes)  # input could be a list
    H_now, W_now = paz.image.get_size(image)
    boxes = paz.boxes.resize_with_aspect_ratio(boxes, H_now, W_now, H, W)
    return boxes


def process_batch(
    key,
    images,
    boxes,
    class_args,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
):
    images = [paz.image.load(image) for image in images]
    iterator = zip(images, boxes)
    boxes = [resize_boxes(*data, H, W) for data in iterator]
    images = [paz.image.resize_with_aspect_ratio(x, H, W) for x in images]
    images = jp.array(images)
    detections = pad(boxes, class_args, max_num_boxes, -1)
    preprocess_images = jax.vmap(paz.lock(preprocess_image, mean), (0, 0))
    keys = jax.random.split(key, len(images))
    images = preprocess_images(keys, images)
    args = prior_boxes, num_classes, match_IOU, variances
    _preprocess_detections = jax.vmap(paz.lock(preprocess_detections, *args))
    detections = _preprocess_detections(detections)
    return images, detections


def compute_num_batches(samples, batch_size):
    return int(math.ceil(samples / batch_size))


class Generator(PyDataset):
    def __init__(
        self, key, images, boxes, class_args, batch_size, pipeline, **kwargs
    ):
        super().__init__(**kwargs)
        if len(images) != len(boxes) != len(class_args):
            raise ValueError("Images, boxes and classes must have same length.")
        self.images = images
        self.boxes = boxes
        self.class_args = class_args
        self.pipeline = pipeline
        self.batch_size = batch_size
        num_batches = compute_num_batches(self.images, batch_size)
        self.keys = jax.random.split(key, num_batches + 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, batch_arg):
        lower_arg = batch_arg * self.batch_size
        upper_arg = min(lower_arg + self.batch_size, len(self.images))
        images = self.images[lower_arg:upper_arg]
        boxes = self.boxes[lower_arg:upper_arg]
        class_args = self.class_args[lower_arg:upper_arg]
        return self.pipeline(self.keys[batch_arg], images, boxes, class_args)

    def on_epoch_end(self):
        # repopulates keys for the next epoch
        self.keys = jax.random.split(self.keys[-1], len(self.images) + 1)
