import jax.numpy as jp
import jax
import paz


def single_augment(key, image, detections, mean, prior_boxes, match_IOU=0.5):
    image = paz.image.augment_color(key, image)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)

    # TODO
    # detections = paz.detection.to_image_coordinates(detections)
    # detections = paz.detection.expand(detections, mean)
    # detections = paz.detection.random_sample_crop(key, detections, 1.0)
    # detections = paz.detection.random_flip_left_right(image, detections)
    # detections = paz.detection.normalized_box_coordinates(
    #     detections, image.shape
    # )

    detections = paz.detection.match(detections, prior_boxes, match_IOU)
    detections = paz.detection.encode(detections)
    detections = paz.detection.to_one_hot(detections)


def pad(boxes, class_args, pad_size, pad_value):

    def pad_sample(boxes, class_args):
        boxes = jp.array(boxes)
        class_args = jp.array(class_args).reshape(-1, 1)
        detections = paz.detection.merge(boxes, class_args)
        detections = paz.detection.pad(detections, pad_size, pad_value)
        return detections

    return jp.array([pad_sample(*sample) for sample in zip(boxes, class_args)])


def preprocess_detections(detections, prior_boxes, match_IOU=0.5):
    detections = paz.detection.match(detections, prior_boxes, match_IOU)
    detections = paz.detection.encode(detections)
    detections = paz.detection.to_one_hot(detections)
    return detections


def preprocess_images(key, image, mean=paz.image.BGR_IMAGENET_MEAN):
    image = paz.image.augment_color(key, image)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    return image


key = jax.random.PRNGKey(0)
images, class_args, boxes = paz.datasets.load("VOC2007", "trainval")
batch_size = 16
batch_arg = 123
lower_arg = batch_arg * batch_size
upper_arg = min(lower_arg + batch_size, len(images))
batch_images = images[lower_arg:upper_arg]
batch_boxes = boxes[lower_arg:upper_arg]
batch_class_args = class_args[lower_arg:upper_arg]
batch_detections = pad(batch_boxes, batch_class_args, 32, -1)
batch_images = jp.array([paz.image.load(image) for image in batch_images])
