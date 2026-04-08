import csv
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from . import kaggle_utils


def load(split="train"):
    split = normalize_split(split)
    csv_path = get_ready_csv_path()
    pixel_strings, class_args = load_split_data(csv_path, split)
    dataset = build_dataset(pixel_strings, class_args)
    return dataset


def download(overwrite=False):
    competition_name = get_competition_name()
    dataset_root = get_dataset_root()
    csv_path = get_csv_path(dataset_root)
    extracted_path = dataset_root / "fer2013" / "fer2013.csv"
    args = competition_name, dataset_root, csv_path
    args += extracted_path, "fer2013.tar.gz", overwrite
    dataset_root = kaggle_utils.download_dataset(*args)
    return dataset_root


def get_class_names():
    return ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def get_competition_name():
    competition_name = "challenges-in-representation-learning-"
    competition_name += "facial-expression-recognition-challenge"
    return competition_name


def get_dataset_root():
    return Path.home() / ".keras" / "paz" / "datasets" / "FER"


def get_csv_path(dataset_root):
    return dataset_root / "fer2013.csv"


def normalize_split(split):
    split = split.lower()
    if split == "val":
        split = "validation"
    return split


def build_split_to_usage():
    split_to_usage = {"train": "Training", "validation": "PublicTest"}
    split_to_usage["test"] = "PrivateTest"
    return split_to_usage


def split_to_usage(split):
    usage = build_split_to_usage()[normalize_split(split)]
    return usage


def get_ready_csv_path():
    dataset_root = get_dataset_root()
    csv_path = get_csv_path(dataset_root)
    if not csv_path.exists():
        dataset_root = download()
        csv_path = get_csv_path(dataset_root)
    return csv_path


def load_split_data(csv_path, split):
    usage = split_to_usage(split)
    pixel_strings, class_args = [], []
    with csv_path.open(newline="") as filedata:
        reader = csv.DictReader(filedata)
        for row in reader:
            if row["Usage"] == usage:
                pixel_strings.append(row["pixels"])
                class_args.append(int(row["emotion"]))
    return pixel_strings, class_args


def build_dataset(pixel_strings, class_args):
    images = build_images(pixel_strings)
    labels = build_labels(class_args)
    dataset = images, labels
    return dataset


def build_images(pixel_strings):
    images = parse_pixel_strings(pixel_strings)
    return images


def build_labels(class_args):
    class_args = jp.array(class_args, dtype=jp.int32)
    labels = jax.nn.one_hot(class_args, 7, dtype=jp.float32)
    return labels


def parse_pixel_strings(pixel_strings):
    num_images = len(pixel_strings)
    pixel_values = np.fromstring(" ".join(pixel_strings), dtype=np.uint8, sep=" ")
    images = jp.array(pixel_values)
    images = jp.reshape(images, (num_images, 48, 48, 1))
    return images


def parse_pixels(pixel_string):
    pixel_values = np.fromstring(pixel_string, dtype=np.uint8, sep=" ")
    image = jp.array(pixel_values)
    image = jp.reshape(image, (48, 48, 1))
    return image
