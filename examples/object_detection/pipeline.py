import jax.numpy as jp
import paz


def augment(key, image, detections, mean, prior_boxes, match_IOU=0.5):
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.augment_color(key, image)
    image = paz.image.subtract_mean(image, mean)
    # paz.detection.pad

    detections = paz.detection.to_image_coordinates(detections)
    detections = paz.detection.expand(detections, mean)
    detections = paz.detection.random_sample_crop(key, detections, 1.0)
    detections = paz.detection.random_flip_left_right(image, detections)
    detections = paz.detection.normalized_box_coordinates(
        detections, image.shape
    )

    detections = paz.detection.match(detections, prior_boxes, match_IOU)
    detections = paz.detection.encode(detections)
    detections = paz.detection.to_one_hot(detections)
