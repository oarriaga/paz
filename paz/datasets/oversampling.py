import numpy as np


def repeat_rare_images(image_paths, detections, rare_class_args, num_copies):
    originals = [(arg, 0) for arg in range(len(image_paths))]
    copies = list_copies(detections, rare_class_args, num_copies)
    image_args, copy_indices = zip(*(originals + copies))
    image_paths = [image_paths[arg] for arg in image_args]
    detections = [detections[arg] for arg in image_args]
    return image_paths, detections, list(copy_indices)


def list_copies(detections, rare_class_args, num_copies):
    image_args = find_rare_images(detections, rare_class_args)
    validate_rare_images(image_args)
    copy_indices = range(1, num_copies + 1)
    return [(arg, index) for arg in image_args for index in copy_indices]


def find_rare_images(detections, rare_class_args):
    is_rare = [has_class(boxes, rare_class_args) for boxes in detections]
    return np.flatnonzero(is_rare).tolist()


def validate_rare_images(image_args):
    if len(image_args) == 0:
        raise ValueError("No image has a box of a rare class")


def has_class(boxes, class_args):
    return bool(np.isin(boxes[:, 4], class_args).any())
