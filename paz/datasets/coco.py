import os
import json
from functools import partial
from operator import itemgetter
from collections import defaultdict

import numpy as np

import paz


def get_class_names():
    return [
        # "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def get_efficientdet_class_names():
    names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
             "train", "truck", "boat", "traffic light", "fire hydrant", "0",
             "stop sign", "parking meter", "bench", "bird", "cat", "dog",
             "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
             "0", "backpack", "umbrella", "0", "0", "handbag", "tie",
             "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
             "baseball bat", "baseball glove", "skateboard", "surfboard",
             "tennis racket", "bottle", "0", "wine glass", "cup", "fork",
             "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
             "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
             "chair", "couch", "potted plant", "bed", "0", "dining table", "0",
             "0", "toilet", "0", "tv", "laptop", "mouse", "remote", "keyboard",
             "cell phone", "microwave", "oven", "toaster", "sink",
             "refrigerator", "0", "book", "clock", "vase", "scissors",
             "teddy bear", "hair drier", "toothbrush"]
    return names


def load_class_names(root):
    data = read_annotations(os.path.join(root, "train"))
    categories = select_classes(data["categories"])
    categories = sorted(categories, key=itemgetter("id"))
    return [category["name"] for category in categories]


def load(root, split, class_names):
    folder = dict(train="train", val="valid", test="test")[split]
    path = os.path.join(root, folder)
    data = read_annotations(path)
    images = sorted(data["images"], key=itemgetter("id"))
    image_paths = [os.path.join(path, image["file_name"]) for image in images]
    detections = parse_detections(data, images, class_names)
    return image_paths, detections


def read_annotations(path):
    json_path = os.path.join(path, "_annotations.coco.json")
    with open(json_path, encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def select_classes(categories):
    return [category for category in categories if is_class(category)]


def is_class(category):
    return category.get("supercategory", "") != "none"


def parse_detections(data, images, class_names):
    category_to_arg = build_category_to_arg(data["categories"], class_names)
    image_to_annotations = group_by_image(data["annotations"])
    parse = partial(parse_image, image_to_annotations, category_to_arg)
    return [parse(image) for image in images]


def build_category_to_arg(categories, class_names):
    name_to_arg = paz.datasets.build_name_to_arg(class_names)
    category_to_arg = {}
    for category in select_classes(categories):
        if category["name"] in name_to_arg:
            category_to_arg[category["id"]] = name_to_arg[category["name"]]
    return category_to_arg


def group_by_image(annotations):
    image_to_annotations = defaultdict(list)
    for annotation in annotations:
        image_to_annotations[annotation["image_id"]].append(annotation)
    return image_to_annotations


def parse_image(image_to_annotations, category_to_arg, image):
    annotations = image_to_annotations[image["id"]]
    annotations = select_annotations(annotations, category_to_arg)
    boxes = read_boxes(annotations, image["width"], image["height"])
    class_args = read_class_args(annotations, category_to_arg)
    detections = np.column_stack([boxes, class_args]).astype("float32")
    return detections[has_area(boxes)]


def select_annotations(annotations, category_to_arg):
    selected = []
    for annotation in annotations:
        if is_selected(annotation, category_to_arg):
            selected.append(annotation)
    return selected


def is_selected(annotation, category_to_arg):
    is_crowd = annotation.get("iscrowd", 0) != 0
    return (not is_crowd) and (annotation["category_id"] in category_to_arg)


def read_boxes(annotations, W, H):
    boxes = [annotation["bbox"] for annotation in annotations]
    boxes = np.array(boxes, "float32").reshape(-1, 4)
    boxes = paz.to_numpy(paz.boxes.xywh_to_xyxy(boxes))
    return np.clip(boxes, 0, np.array([W, H, W, H], "float32"))


def read_class_args(annotations, category_to_arg):
    category_ids = [annotation["category_id"] for annotation in annotations]
    return [category_to_arg[category_id] for category_id in category_ids]


def has_area(boxes):
    return np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
