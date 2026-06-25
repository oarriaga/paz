import os
from glob import glob

import numpy as np

import paz


def get_class_names():
    return ["void", "flat", "construction", "object",
            "nature", "sky", "human", "vehicle"]


CATEGORY_RANGES = [(0, 6), (7, 10), (11, 16), (17, 20),
                   (21, 22), (23, 23), (24, 25), (26, 33)]


def build_id_to_category():
    table = np.zeros(34, "int32")
    for category, (low, high) in enumerate(CATEGORY_RANGES):
        table[low:high + 1] = category
    return table


ID_TO_CATEGORY = build_id_to_category()


def validate_split(split):
    assert split in ("train", "val", "test")


def to_label_path(image_path):
    folder = os.path.dirname(image_path).replace("leftImg8bit", "gtFine")
    name = os.path.basename(image_path)
    name = name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    return os.path.join(folder, name)


def load(root, split="train"):
    validate_split(split)
    pattern = os.path.join(root, "leftImg8bit", split, "*", "*_leftImg8bit.png")
    image_paths = sorted(glob(pattern))
    label_paths = [to_label_path(path) for path in image_paths]
    for label_path in label_paths:
        assert os.path.exists(label_path), f"Missing label {label_path}"
    return image_paths, label_paths


def load_image(path, size):
    return paz.image.resize_opencv(paz.image.load(path), size)


def load_mask(path, size):
    label = paz.image.load(path, paz.image.GRAY)
    label = paz.image.resize(label, size, "nearest")
    label = np.clip(np.asarray(label)[..., 0], 0, 33)
    return ID_TO_CATEGORY[label]
