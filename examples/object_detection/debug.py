# set negatives with -1 in match to remove extra background class
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
from pipeline import AugmentDetection


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
max_num_boxes = 32

lower_arg = batch_arg * batch_size
upper_arg = min(lower_arg + batch_size, len(images))

batch_images = images[lower_arg:upper_arg]
batch_boxes = boxes[lower_arg:upper_arg]
batch_class_args = class_args[lower_arg:upper_arg]

x_true, y_true = AugmentDetection(
    key,
    batch_images,
    batch_boxes,
    batch_class_args,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
)


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
    one_hot_vectors = paz.detection.get_scores(detections)
    detections = to_class_args(detections)
    boxes, class_args = paz.detection.split(detections)
    mask = class_args != class_arg
    boxes = jp.where(mask, boxes, value)
    class_args = jp.where(mask, one_hot_vectors, value)
    detections = paz.detection.merge(boxes, one_hot_vectors)
    return paz.detection.remove_class(detections, class_arg)


def deprocess_detections(detections, prior_boxes, variances):
    detections = paz.detection.decode(detections, prior_boxes, variances)
    detections = filter_class_arg(detections, 0)
    detections = paz.detection.denormalize(detections, 300, 300)
    return to_boxes2D(detections)


def to_boxes2D(detections):
    boxes, one_hot_vectors = paz.detection.split(detections)
    class_args = jp.argmax(one_hot_vectors, axis=-1, keepdims=False)
    scores = jp.max(one_hot_vectors, axis=-1, keepdims=False)
    return boxes.astype("int32"), class_args.astype("int32"), scores


x = deprocess_image(x_true[0], mean)
y_boxes, y_class_args, y_scores = deprocess_detections(
    y_true[0], prior_boxes, variances
)
paz.image.show(
    paz.draw.boxes2D(
        x,
        y_boxes,
        y_class_args,
        y_scores,
        names=paz.datasets.labels("VOC"),
        colors=paz.draw.lincolor(num_classes),
    )
)


# image_path = images[0]
# image = paz.image.load(image_path)
# H, W = paz.image.get_size(image)
# image_boxes = jp.array(boxes[0])
# image_with_boxes = paz.draw.boxes(image, image_boxes)
# paz.image.show(image_with_boxes)

# resized_image = paz.image.resize(image, (300, 300))
# resized_boxes = paz.boxes.resize(image_boxes, H, W, 300, 300)
# image_with_resized_boxes = paz.draw.boxes(resized_image, resized_boxes)
# paz.image.show(image_with_resized_boxes.astype("uint8"))


# def build_negative_mask(detections):
#     is_invalid = jp.any(detections < 0.0, axis=1)
#     return is_invalid


# def build_positive_mask(detections):
#     is_invalid = jp.any(detections < 0.0, axis=1)
#     is_valid = jp.logical_not(is_invalid)
#     return is_valid
