# all resizes should output an image in uint8 format
# TODO the problem is that the boxes need to be resize as well.
# detections = paz.detection.to_image_coordinates(detections)
# detections = paz.detection.expand(detections, mean)
# detections = paz.detection.random_sample_crop(key, detections, 1.0)
# detections = paz.detection.random_flip_left_right(image, detections)
# detections = paz.detection.normalized_box_coordinates(
#     detections, image.shape
# )
from jax import random
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


def deprocess_image(image, mean):
    image = image + mean
    image = paz.image.BGR_to_RGB(image)
    image = paz.cast(image, "uint8")
    return image


def to_class_args(detections):
    boxes, one_hot_vectors = paz.detection.split(detections)
    class_args = jp.argmax(one_hot_vectors, axis=-1, keepdims=True)
    return paz.detection.merge(boxes, class_args)


def filter_class_arg(detections, class_arg, value=-1):
    detections = to_class_args(detections)
    boxes, class_args = paz.detection.split(detections)
    mask = class_args != class_arg
    boxes = jp.where(mask, boxes, value)
    class_args = jp.where(mask, class_args, value)
    return paz.detection.merge(boxes, class_args)


def build_negative_mask(detections):
    is_invalid = jp.any(detections < 0.0, axis=1)
    return is_invalid


def build_positive_mask(detections):
    is_invalid = jp.any(detections < 0.0, axis=1)
    is_valid = jp.logical_not(is_invalid)
    return is_valid


def to_boxes2D(detections):
    # negative_mask = build_negative_mask(detections)
    print("dets shape", detections.shape)
    boxes, one_hot_vectors = paz.detection.split(detections)
    class_args = jp.argmax(one_hot_vectors, axis=-1, keepdims=False)
    scores = jp.max(one_hot_vectors, axis=-1, keepdims=False)
    return boxes.astype("int32"), class_args.astype("int32"), scores


def deprocess_detections(detections, prior_boxes, variances):
    detections = paz.detection.decode(detections, prior_boxes, variances)
    # detections = paz.detection.remove_class(detections, 0)
    detections = filter_class_arg(detections, 0)
    detections = paz.detection.denormalize(detections, 300, 300)
    return to_boxes2D(detections)


def resize_boxes(image, boxes, H, W):
    boxes = jp.array(boxes)  # input could be a list
    H_now, W_now = paz.image.get_size(image)
    boxes = paz.boxes.resize(boxes, H_now, W_now, H, W)
    return boxes


key = random.PRNGKey(0)
images, class_args, boxes = paz.datasets.load("VOC2007", "trainval")
batch_size = 16
match_IOU = 0.5
mean = jp.array(paz.image.BGR_IMAGENET_MEAN)
prior_boxes = paz.models.detection.utils.create_prior_boxes("VOC")
variances = [0.1, 0.1, 0.2, 0.2]
batch_arg = 123
num_classes = 20
H = W = 300
lower_arg = batch_arg * batch_size
upper_arg = min(lower_arg + batch_size, len(images))
batch_images = images[lower_arg:upper_arg]
batch_boxes = boxes[lower_arg:upper_arg]
batch_class_args = class_args[lower_arg:upper_arg]
batch_images = [paz.image.load(image) for image in batch_images]


batch_boxes = [
    resize_boxes(*data, H, W) for data in zip(batch_images, batch_boxes)
]

batch_images = [
    # paz.image.resize_with_aspect_ratio(image, H, W) for image in batch_images
    paz.image.resize(image, (H, W))
    for image in batch_images
]


batch_detections = pad(batch_boxes, batch_class_args, 32, -1)
batch_images = jp.array(batch_images)
preprocess_images = jax.vmap(paz.lock(preprocess_image, mean), in_axes=(0, 0))
batch_images = preprocess_images(random.split(key, batch_size), batch_images)
# [paz.image.show(deprocess_image(image, mean)) for image in batch_images]
args = prior_boxes, num_classes, match_IOU, variances
vpreprocess_detections = jax.vmap(paz.lock(preprocess_detections, *args))
vbatch_detections = vpreprocess_detections(batch_detections)

x = deprocess_image(batch_images[0], mean)
y_boxes, y_class_args, y_scores = deprocess_detections(
    vbatch_detections[0], prior_boxes, variances
)
# filter instead of removing class
image_with_boxes = paz.draw.boxes2D(
    x,
    y_boxes,
    y_class_args,
    y_scores,
    names=paz.datasets.labels("VOC"),
    colors=paz.draw.lincolor(num_classes),
)
paz.image.show(image_with_boxes)


image_path = images[0]
image = paz.image.load(image_path)
H, W = paz.image.get_size(image)
image_boxes = jp.array(boxes[0])
image_with_boxes = paz.draw.boxes(image, image_boxes)
paz.image.show(image_with_boxes)

resized_image = paz.image.resize(image, (300, 300))
resized_boxes = paz.boxes.resize(image_boxes, H, W, 300, 300)
image_with_resized_boxes = paz.draw.boxes(resized_image, resized_boxes)
paz.image.show(image_with_resized_boxes.astype("uint8"))
