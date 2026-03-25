import os
from collections import namedtuple
from glob import glob
from pathlib import Path
import pickle

from keras.utils import get_file
import cv2
import numpy as np
import jax.numpy as jp
import paz

Shot = namedtuple(
    "Shot", ["image", "depth", "masks", "boxes", "pointcloud", "label"]
)


def download():
    URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.21/CLEVR-POSE.zip"  # fmt: skip
    directory, filename = "paz/datasets/", "CLEVR-POSE"
    path = Path(get_file(filename, URL, cache_subdir=directory, extract=True))
    return path / filename


def get_y_FOV():
    return 0.742


def get_intrinsics():
    return jp.array(
        [[616.94434, 0.0, 320.0], [0.0, 616.94434, 240.0], [0.0, 0.0, 1.0]]
    )


def load_depth(filepath):
    return np.expand_dims(np.load(filepath), axis=2)


def load_label(filepath):
    return pickle.load(open(filepath, "rb"))


def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask.astype(bool)


def extract_masks(mask_image):
    masks = []
    unique_mask_values = np.unique(mask_image // 20)
    for value in range(len(unique_mask_values) - 1):
        mask = (mask_image == (value + 1) * 20).astype(np.uint8)
        masks.append(clean_mask(mask))
    return jp.array(masks)


def load_masks(filepath):
    masks = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    masks = extract_masks(masks)
    return masks


def masks_to_boxes(masks):
    boxes = [paz.mask.to_box(mask.astype(float), 1.0) for mask in masks]
    boxes = paz.boxes.xyxy_to_xywh(jp.array(boxes))
    return np.array(boxes)


def load_shot(directory):
    label = load_label(os.path.join(directory, "label.pkl"))
    image = paz.image.load(os.path.join(directory, "image.png"))
    depth = load_depth(os.path.join(directory, "depth.npy"))
    masks = load_masks(os.path.join(directory, "masks.png"))
    boxes = np.array(masks_to_boxes(masks).astype(int))
    masks = np.expand_dims(np.array(masks), axis=-1)
    pointcloud = paz.pointcloud.from_depth(depth, get_intrinsics())
    return Shot(image, depth, masks, boxes, pointcloud, label)


def load(concept_arg):
    root = download()
    concept_path = sorted(root.glob("scenes/*"))[concept_arg]
    return [load_shot(concept_path)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def draw_rectangle(image, corner_A, corner_B, color, thickness):
        return cv2.rectangle(
            image, tuple(corner_A), tuple(corner_B), tuple(color), thickness
        )

    def draw_center_form_box(image, box, color=[0, 255, 0], thick=2):
        image = paz.to_numpy(image)
        x_min, y_min, W_box, H_box = box.tolist()
        x_max = x_min + W_box
        y_max = y_min + H_box
        corner_A, corner_B = (x_min, y_min), (x_max, y_max)
        draw_rectangle(image, corner_A, corner_B, color, thick)
        image = paz.to_jax(image)
        return image

    def plot_shot(shot):
        num_masks = len(shot.masks)
        figure, axes = plt.subplots(1, 1 + num_masks)
        image = shot.image.copy()
        for box in shot.boxes:
            image = draw_center_form_box(image, box)
        axes[0].imshow(image)
        for mask_arg in range(num_masks):
            axes[mask_arg + 1].imshow((shot.masks[mask_arg]))
        plt.show()

    for concept_arg in range(2):
        plot_shot(load(concept_arg)[shot_arg := 0])
