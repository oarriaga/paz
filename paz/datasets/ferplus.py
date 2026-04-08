import csv
from pathlib import Path
from urllib.request import urlretrieve

import jax.numpy as jp

from . import fer


def load(split="train", image_shape=(48, 48)):
    split = fer.normalize_split(split)
    csv_path, labels_path = get_ready_paths()
    label_by_index = load_probability_labels(labels_path, split)
    dataset = build_dataset(csv_path, label_by_index, split, image_shape)
    return dataset


def download(overwrite=False):
    dataset_root = fer.download(overwrite=overwrite)
    labels_path = get_labels_path(dataset_root)
    if labels_path.exists() and not overwrite:
        return dataset_root
    try:
        urlretrieve(get_labels_url(), labels_path)
    except Exception as error:
        raise RuntimeError(build_download_message()) from error
    return dataset_root


def get_class_names():
    class_names = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt",
    ]
    return class_names


def get_labels_url():
    labels_url = "https://raw.githubusercontent.com/microsoft/FERPlus/"
    labels_url += "master/fer2013new.csv"
    return labels_url


def get_dataset_root():
    dataset_root = fer.get_dataset_root()
    return dataset_root


def get_labels_path(dataset_root):
    labels_path = dataset_root / "fer2013new.csv"
    return labels_path


def get_ready_paths():
    dataset_root = get_dataset_root()
    csv_path = fer.get_csv_path(dataset_root)
    labels_path = get_labels_path(dataset_root)
    if not csv_path.exists() or not labels_path.exists():
        dataset_root = download()
        csv_path = fer.get_csv_path(dataset_root)
        labels_path = get_labels_path(dataset_root)
    return csv_path, labels_path


def load_probability_labels(filepath, split):
    usage = fer.split_to_usage(split)
    label_by_index = {}
    with Path(filepath).open(newline="") as filedata:
        reader = csv.DictReader(filedata)
        for row in reader:
            if row["Usage"] == usage:
                image_index = parse_image_index(row["Image name"])
                votes = parse_votes(row)
                if float(jp.sum(votes)) != 0.0:
                    label_by_index[image_index] = normalize_votes(votes)
    return label_by_index


def build_dataset(csv_path, label_by_index, split, image_shape):
    usage = fer.split_to_usage(split)
    images, labels = [], []
    with csv_path.open(newline="") as filedata:
        reader = csv.DictReader(filedata)
        for row_index, row in enumerate(reader):
            is_valid_usage = row["Usage"] == usage
            has_label = row_index in label_by_index
            if is_valid_usage and has_label:
                image = fer.parse_pixels(row["pixels"])
                image = fer.resize_image(image, image_shape)
                images.append(image)
                labels.append(label_by_index[row_index])
    images = jp.stack(images)
    labels = jp.stack(labels)
    dataset = images, labels
    return dataset


def parse_votes(row):
    votes = [float(row[class_name]) for class_name in get_class_names()]
    votes = jp.array(votes, dtype=jp.float32)
    return votes


def normalize_votes(votes):
    num_votes = jp.sum(votes)
    normalized_votes = votes / num_votes
    return normalized_votes


def parse_image_index(image_name):
    stem = Path(image_name).stem
    if not stem.startswith("fer"):
        raise ValueError(f"Invalid FERPlus image name: {image_name}")
    image_index = int(stem.removeprefix("fer"))
    return image_index


def build_download_message():
    message = "Failed to download FERPlus labels from "
    message += get_labels_url()
    return message
