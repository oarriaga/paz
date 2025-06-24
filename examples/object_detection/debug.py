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
from pipeline2 import preprocess_batch


key = random.PRNGKey(0)
# images, detections = paz.datasets.load("VOC2007", "trainval")
images, detections = paz.datasets.deepfish.load("train")
batch_size = 16
match_IOU = 0.5
mean = jp.array(paz.image.BGR_IMAGENET_MEAN)
prior_boxes = paz.models.detection.single_shot_detector.build_prior_boxes("VOC")
variances = [0.1, 0.1, 0.2, 0.2]
batch_arg = 123
num_classes = 20
H = W = 300
max_num_boxes = 32

lower_arg = batch_arg * batch_size
upper_arg = min(lower_arg + batch_size, len(images))

batch_images = images[lower_arg:upper_arg]
batch_labels = detections[lower_arg:upper_arg]

x_true, y_true = preprocess_batch(
    key,
    batch_images,
    batch_labels,
    H,
    W,
    prior_boxes,
    num_classes,
    match_IOU,
    variances,
    mean,
    max_num_boxes,
    True,
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
    is_not_class_arg = class_args != class_arg
    boxes = jp.where(is_not_class_arg, boxes, value)
    class_args = jp.where(is_not_class_arg, one_hot_vectors, value)
    detections = paz.detection.merge(boxes, one_hot_vectors)
    return paz.detection.remove_class(detections, class_arg)


def deprocess_detections(detections, prior_boxes, variances):
    detections = paz.detection.decode(detections, prior_boxes, variances)
    detections = filter_class_arg(detections, 0)
    detections = paz.detection.denormalize(detections, 300, 300)
    return to_boxes2D(detections)


def select_prior_boxes(detections, prior_boxes, variances, class_arg=0):
    detections = paz.detection.decode(detections, prior_boxes, variances)
    one_hot_vectors = paz.detection.get_scores(detections)
    detections = to_class_args(detections)
    boxes, class_args = paz.detection.split(detections)
    is_not_class_arg = class_args != class_arg
    prior_boxes = paz.boxes.to_corner_form(prior_boxes)
    boxes = jp.where(is_not_class_arg, prior_boxes, -1)
    class_args = jp.where(is_not_class_arg, one_hot_vectors, -1)
    detections = paz.detection.merge(boxes, one_hot_vectors)
    detections = paz.detection.remove_class(detections, class_arg)
    detections = paz.detection.denormalize(detections, 300, 300)
    return to_boxes2D(detections)


def to_boxes2D(detections):
    boxes, one_hot_vectors = paz.detection.split(detections)
    class_args = jp.argmax(one_hot_vectors, axis=-1, keepdims=False)
    scores = jp.max(one_hot_vectors, axis=-1, keepdims=False)
    return boxes.astype("int32"), class_args.astype("int32"), scores


def fn_pos(y_true_sample):
    return y_true_sample[y_true_sample[:, 4] != 1]


def fn_neg(y_true_sample):
    return y_true_sample[y_true_sample[:, 4] == 1]


sample_arg = 0
x = deprocess_image(x_true[sample_arg], mean)
y_boxes, y_class_args, y_scores = deprocess_detections(
    y_true[sample_arg], prior_boxes, variances
)

paz.image.show(
    paz.draw.boxes2D(
        x,
        y_boxes,
        y_class_args,
        y_scores,
        names=paz.datasets.labels("VOC"),
        colors=paz.draw.lincolor(num_classes),
        thickness=1,
        font_scale=0.5,
    )
)


y_boxes, y_class_args, y_scores = select_prior_boxes(
    y_true[sample_arg], prior_boxes, variances
)
paz.image.show(
    paz.draw.boxes2D(
        x,
        y_boxes,
        y_class_args,
        y_scores,
        names=paz.datasets.labels("VOC"),
        colors=paz.draw.lincolor(num_classes),
        thickness=2,
        font_scale=0.0,
    )
)

print(fn_pos(y_true[sample_arg]))
print(fn_neg(y_true[sample_arg]))
# detections = paz.detection.normalize(detections, H, W)
# detections = add_background_class(detections)
# detections = paz.detection.match(detections, prior_boxes, IOU)
