import csv
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import paz

from . import kaggle_utils


def load(split="train", image_shape=(48, 48)):
    split = normalize_split(split)
    csv_path = get_ready_csv_path()
    rows = load_rows(csv_path, split)
    dataset = build_dataset(rows, image_shape)
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


def load_rows(csv_path, split):
    usage = split_to_usage(split)
    rows = []
    with csv_path.open(newline="") as filedata:
        reader = csv.DictReader(filedata)
        for row in reader:
            if row["Usage"] == usage:
                rows.append(row)
    return rows


def build_dataset(rows, image_shape):
    pixel_strings, class_args = unpack_rows(rows)
    images = build_images(pixel_strings, image_shape)
    labels = build_labels(class_args)
    dataset = images, labels
    return dataset


def unpack_rows(rows):
    pixel_strings, class_args = [], []
    for row in rows:
        pixel_strings.append(row["pixels"])
        class_args.append(int(row["emotion"]))
    return pixel_strings, class_args


def build_images(pixel_strings, image_shape):
    images = [parse_pixels(pixel_string) for pixel_string in pixel_strings]
    images = jp.stack(images)
    images = resize_images(images, image_shape)
    return images


def resize_images(images, image_shape):
    resize_batch = jax.vmap(resize_image, in_axes=(0, None))
    resize_batch = jax.jit(resize_batch, static_argnums=1)
    resized_images = resize_batch(images, image_shape)
    return resized_images


@jax.jit
def build_labels(class_args):
    class_args = jp.array(class_args, dtype=jp.int32)
    labels = jax.nn.one_hot(class_args, 7, dtype=jp.float32)
    return labels


def parse_pixels(pixel_string):
    pixel_values = np.fromstring(pixel_string, dtype=np.uint8, sep=" ")
    image = jp.array(pixel_values)
    image = jp.reshape(image, (48, 48, 1))
    return image


def resize_image(image, image_shape):
    resized_image = paz.image.resize(image, image_shape)
    resized_image = paz.cast(resized_image, jp.uint8)
    return resized_image
