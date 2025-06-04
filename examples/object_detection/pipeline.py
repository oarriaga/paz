import jax.numpy as jp
import paz


def augment(key, image, detections, mean, augment=True):
    image = paz.image.RGB_to_BGR(image)
    if augment:
        image = paz.image.augment_color(key, image)
    image = paz.image.subtract_mean(image, mean)
    # paz.detection.pad
    # paz.boxes.pad
    detections = paz.detection.encode(detections)
    detections = paz.detection.to_one_hot(detections)
