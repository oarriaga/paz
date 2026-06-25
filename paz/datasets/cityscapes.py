import os
from glob import glob

import numpy as np

import paz
from paz.datasets import kaggle_utils


CITYSCAPES_PACKAGES = ("gtFine_trainvaltest.zip",
                       "leftImg8bit_trainvaltest.zip")


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


def download(root, packages=CITYSCAPES_PACKAGES):
    downloader = build_downloader()
    os.makedirs(root, exist_ok=True)
    session = downloader.login()
    downloader.download_packages(session, list(packages), root, resume=True)
    for package in packages:
        kaggle_utils.extract_archive(os.path.join(root, package), root)
    return root


def build_downloader():
    try:
        from cityscapesscripts.download import downloader
    except ImportError as error:
        raise ImportError(build_download_message()) from error
    return downloader


def build_download_message():
    return ("Cityscapes is login-gated. Run `pip install cityscapesScripts` "
            "and export CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD from a "
            "free account at https://www.cityscapes-dataset.com to download.")


def download_kaggle(root, dataset="xiaose/cityscapes"):
    api = kaggle_utils.build_api()
    os.makedirs(root, exist_ok=True)
    api.dataset_download_files(dataset, path=root, unzip=True)
    return adapt_kaggle_layout(root)


def adapt_kaggle_layout(root):
    matches = glob(os.path.join(root, "**", "gtFine"), recursive=True)
    assert matches, f"No gtFine directory found under {root}"
    base = os.path.dirname(matches[0])
    images = os.path.join(base, "leftImg8bit")
    if not os.path.exists(images):
        os.symlink("images", images)
    return base


def load_image(path, size):
    return paz.image.resize_opencv(paz.image.load(path), size)


def load_mask(path, size):
    label = paz.image.load(path, paz.image.GRAY)
    label = paz.image.resize(label, size, "nearest")
    label = np.clip(np.asarray(label)[..., 0], 0, 33)
    return ID_TO_CATEGORY[label]
