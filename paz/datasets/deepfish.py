import os
import glob

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


def parse_line(line):
    values = [float(value) for value in line.strip().split()]
    class_arg, x_center, y_center, box_W, box_H = values
    x_min = x_center - (box_W / 2)
    x_max = x_center + (box_W / 2)
    y_min = y_center - (box_H / 2)
    y_max = y_center + (box_H / 2)
    x_min = int(x_min * 1920)
    y_min = int(y_min * 1080)
    x_max = int(x_max * 1920)
    y_max = int(y_max * 1080)
    return x_min, y_min, x_max, y_max, class_arg


def parse_file(filepath):
    detections = []
    with open(filepath, "r") as file:
        detections.extend([parse_line(line) for line in file])
    return detections


def load(split="train"):
    if split not in ["train", "validation"]:
        raise ValueError("Valid splits include 'train' or 'validation'")
    split = "valid" if split == "validation" else split
    path = download(overwrite=False)
    images = sorted(glob.glob(f"{path}/*/{split}/*.jpg"))
    filepaths = sorted(glob.glob(f"{path}/*/{split}/*.txt"))
    detections = [parse_file(filepath) for filepath in filepaths]
    return images, detections


def get_class_names():
    return ["fish"]
