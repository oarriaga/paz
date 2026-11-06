import os
import re
import json
import random
from pathlib import Path, PurePosixPath
from operator import itemgetter

import numpy as np

import paz


IMAGE_EXTENSIONS = re.compile(r"\.(jpe?g|png|bmp|tiff?)", re.IGNORECASE)


def load(root, split):
    validate_split(split)
    samples = read_samples(root)
    validate_samples(samples, root)
    samples = select_split(samples, split)
    image_paths = [image_path for image_path, _ in samples]
    detections = [parse_sample(*sample) for sample in samples]
    return image_paths, detections


def validate_split(split):
    if split not in ["train", "validation"]:
        raise ValueError("Valid splits include 'train' or 'validation'")


def validate_samples(samples, root):
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 OzFish samples under {root}")


def select_split(samples, split):
    # seed and train fraction of the paper's OzFish split
    shuffled = list(samples)
    random.Random(42).shuffle(shuffled)
    num_train = min(max(1, round(len(shuffled) * 0.8)), len(shuffled) - 1)
    train, validation = shuffled[:num_train], shuffled[num_train:]
    return validation if split == "validation" else train


def read_samples(root):
    image_index = build_image_index(os.path.join(root, "frames_labelled"))
    rows = read_manifests(os.path.join(root, "manifests"))
    validate_keys(rows, image_index)
    return [(image_index[key], payload) for key, payload in rows]


def build_image_index(frames_path):
    image_index = {}
    for image_path in list_sorted_files(frames_path, "**/*"):
        if has_image_suffix(image_path):
            image_index.setdefault(cut_key(image_path), image_path)
    return image_index


def read_manifests(manifests_path):
    rows = []
    for manifest_path in list_sorted_files(manifests_path, "*"):
        rows.extend(read_rows(manifest_path))
    return rows


def validate_keys(rows, image_index):
    missing_keys = [key for key, _ in rows if key not in image_index]
    if missing_keys:
        raise FileNotFoundError(f"OzFish frames not found: {missing_keys}")


def list_sorted_files(path, pattern):
    files = paz.directory.list_files(path, pattern)
    # part by part and case-sensitive, as pathlib sorts on Linux
    return sorted(files, key=lambda file_path: Path(file_path).parts)


def has_image_suffix(path):
    return bool(IMAGE_EXTENSIONS.fullmatch(os.path.splitext(path)[1]))


def cut_key(image_path):
    name = os.path.basename(image_path)
    return name[:IMAGE_EXTENSIONS.search(name).end()]


def read_rows(manifest_path):
    rows = []
    for record in read_records(manifest_path):
        payloads = [value for value in record.values() if is_payload(value)]
        if payloads:
            rows.append((read_key(record), payloads[0]))
    return rows


def read_records(manifest_path):
    with open(manifest_path, encoding="utf-8") as manifest_file:
        lines = manifest_file.readlines()
    return [json.loads(line) for line in lines if line.strip()]


def is_payload(value):
    is_dict = isinstance(value, dict)
    return is_dict and ("annotations" in value) and ("image_size" in value)


def read_key(record):
    return PurePosixPath(record.get("source-ref", "")).name


def parse_sample(image_path, payload):
    H, W = paz.image.get_size(paz.image.load(image_path))
    boxes = parse_boxes(payload, W, H).astype("float32")
    boxes = boxes[has_area(boxes)]
    return paz.to_numpy(paz.boxes.append_class(boxes, 0))


def parse_boxes(payload, W, H):
    read_box = itemgetter("left", "top", "width", "height")
    boxes = [read_box(annotation) for annotation in payload["annotations"]]
    scale = compute_scale(payload["image_size"][0], W, H)
    boxes = to_corners(np.array(boxes, "float64").reshape(-1, 4) * scale)
    return np.clip(boxes, 0, [W, H, W, H])


def compute_scale(image_size, W, H):
    manifest_W = image_size.get("width") or W
    manifest_H = image_size.get("height") or H
    x_scale, y_scale = W / manifest_W, H / manifest_H
    return np.array([x_scale, y_scale, x_scale, y_scale])


def to_corners(boxes):
    x_min_y_min = boxes[:, :2]
    return np.concatenate([x_min_y_min, x_min_y_min + boxes[:, 2:]], axis=1)


def has_area(boxes):
    return np.all(boxes[:, 2:] > boxes[:, :2], axis=1)


def get_class_names():
    return ["fish"]
