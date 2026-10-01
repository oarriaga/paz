import os
import glob

import numpy as np

import paz


def download(overwrite=False):
    try:
        import gdown
    except ImportError as error:
        print(f"Import Error: {error}")
        raise ImportError("Please install gdown: pip install gdown")

    GDRIVE_ID = "10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0"
    root = os.path.expanduser("~/.keras/paz/datasets/")
    zip_filepath = os.path.join(root, "Deepfish.zip")
    extracted_path = os.path.join(root, "Deepfish")

    if os.path.exists(extracted_path) and not overwrite:
        print(f"Dataset already found at: {extracted_path}")
        return extracted_path

    print(f"Creating directory: {root}")
    os.makedirs(root, exist_ok=True)
    gdown.download(id=GDRIVE_ID, output=zip_filepath, quiet=False)
    paz.utils.extract(zip_filepath)
    if os.path.exists(zip_filepath):
        print(f"Removing temporary file: {zip_filepath}")
        os.remove(zip_filepath)
    return extracted_path


def load(split="train"):
    validate_split(split)
    folder = "valid" if split == "validation" else split
    root = download(overwrite=False)
    image_paths = sorted(glob.glob(os.path.join(root, "*", folder, "*.jpg")))
    return image_paths, parse_images(image_paths)


def load_random_split(split):
    validate_split(split)
    root = download(overwrite=False)
    pattern = os.path.join(root, "*", "*", "*.jpg")
    image_paths = sorted(glob.glob(pattern), key=to_posix_path)
    image_paths = select_split(image_paths, split)
    return image_paths, parse_images(image_paths)


def validate_split(split):
    if split not in ["train", "validation"]:
        raise ValueError("Valid splits include 'train' or 'validation'")


def to_posix_path(path):
    return path.replace(os.sep, "/")


def select_split(image_paths, split):
    # seed and validation fraction of the paper's pooled 80/20 split
    indices = np.random.RandomState(42).permutation(len(image_paths))
    num_val = max(1, int(len(image_paths) * 0.2))
    validation, train = np.split(indices, [num_val])
    split_indices = validation if split == "validation" else train
    return [image_paths[index] for index in np.sort(split_indices)]


def parse_images(image_paths):
    return [parse_image(image_path) for image_path in image_paths]


def parse_image(image_path):
    H, W = paz.image.get_size(paz.image.load(image_path))
    label_path = os.path.splitext(image_path)[0] + ".txt"
    boxes = parse_boxes(read_lines(label_path), W, H).astype("float32")
    boxes = boxes[has_area(boxes)]
    return paz.to_numpy(paz.boxes.append_class(boxes, 0))


def read_lines(label_path):
    if os.path.isfile(label_path):
        with open(label_path) as label_file:
            lines = label_file.readlines()
    else:
        lines = []
    return lines


def parse_boxes(lines, W, H):
    boxes = [parse_line(line, W, H) for line in lines if line.strip()]
    boxes = np.array(boxes).reshape(-1, 4)
    return np.clip(boxes, 0, [W, H, W, H])


def parse_line(line, W, H):
    _, x_center, y_center, box_W, box_H = map(float, line.split())
    box_W, box_H = box_W * W, box_H * H
    x_min = x_center * W - box_W / 2
    y_min = y_center * H - box_H / 2
    return [x_min, y_min, x_min + box_W, y_min + box_H]


def has_area(boxes):
    return np.all(boxes[:, 2:] > boxes[:, :2], axis=1)


def get_class_names():
    return ["fish"]
