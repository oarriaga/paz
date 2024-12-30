from functools import partial
from collections import namedtuple
import numpy as np

import paz


Shape = namedtuple("Shape", ["name", "color", "box", "center", "size"])


def get_class_names():
    return ["square", "circle", "triangle"]


def build_shape(H, W, class_names, offset=20):
    name = np.random.choice(class_names)
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    center_x = np.random.randint(offset, W - offset - 1)
    center_y = np.random.randint(offset, H - offset - 1)
    size = np.random.randint(offset, H // 4)
    box = build_bounding_box(center_x, center_y, size)
    return Shape(name, color, box, (center_x, center_y), size)


def build_bounding_box(center_x, center_y, size):
    x_min, y_min = center_x - size, center_y - size
    x_max, y_max = center_x + size, center_y + size
    return [x_min, y_min, x_max, y_max]


def build_background(H, W):
    background_color = np.random.randint(0, 255, size=3)
    image = np.full([H, W, 3], background_color, np.uint8)
    return image


def draw_shapes(H, W, shapes):
    image = build_background(H, W)
    for shape in shapes:
        image = draw_shape(image, shape)
    return image


def draw_shape(image, shape):
    args = (image, shape.center, shape.size, shape.color)
    if shape.name == "square":
        image = paz.backend.draw.square(*args)
    elif shape.name == "circle":
        image = paz.backend.draw.circle(*args)
    elif shape.name == "triangle":
        image = paz.backend.draw.triangle(*args)
    else:
        raise ValueError(f"Invalid class name {shape.name}")
    return image


def draw_masks(H, W, shapes, name_to_arg):
    mask = np.zeros([H, W, 1])
    for shape in shapes:
        class_arg = name_to_arg[shape.name] + 1  # background is zero
        mask = draw_shape(mask, shape, (class_arg, class_arg, class_arg))
    return mask.astype(np.uint8)


def remove_overlaps(shapes, IOU_thresh):
    boxes = np.array([shape.box for shape in shapes])
    scores = np.ones(len(boxes))
    args, num_boxes = paz.detection.non_max_suppression(
        boxes, scores, IOU_thresh)
    selected_args = args[:num_boxes]
    return [shapes[arg] for arg in selected_args]


def sample(H, W, iou_thresh, num_shapes, arg_to_name):
    shapes = [build_shape(H, W) for _ in range(num_shapes)]
    shapes = remove_overlaps(shapes, iou_thresh)
    image = draw_shapes(H, W, shapes)
    masks = draw_masks(H, W, shapes)
    boxes = [shape.box for shape in shapes]
    class_args = [arg_to_name.index(shape.name) for shape in shapes]
    return image, class_args, boxes, masks


def load(H, W, iou_thresh, max_num_shapes, num_samples):
    images, class_args, masks, boxes = [], [], [], []
    _sample = partial(sample, H, W, iou_thresh, max_num_shapes)
    for _ in range(num_samples):
        image, classes, sample_boxes, sample_masks = _sample()
        images.append(image)
        boxes.append(sample_boxes)
        masks.append(sample_masks)
        class_args.append(classes)
    return images, masks, boxes, class_args
