import os
from pathlib import Path
from keras.utils import get_file
from glob import glob
from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jp
import paz
from scipy.io import loadmat


ShotFields = ["image", "depth", "masks", "boxes", "pointcloud", "label"]
Shot = namedtuple("Shot", ShotFields)


def download():
    dataset_root = get_dataset_root()
    return ensure_dataset(dataset_root)


def get_dataset_root():
    return Path.home() / ".keras" / "paz" / "datasets" / "FEWSOL" / "real_objects"


def ensure_dataset(dataset_root):
    if dataset_root.exists():
        return dataset_root
    url = "https://github.com/oarriaga/altamira-data/releases/download/v0.20/"
    url = url + "real_objects.zip"
    path = get_file("FEWSOL", url, cache_subdir="paz/datasets/", extract=True)
    return Path(path) / "real_objects"


def get_markers():
    marker_to_position = {
        "marker_00": [2.9500, 2.95],
        "marker_01": [15.850, 2.95],
        "marker_02": [28.750, 2.95],
        "marker_03": [41.550, 2.95],
        "marker_04": [54.450, 2.95],
        "marker_05": [67.450, 2.95],
        "marker_06": [67.45, 13.75],
        "marker_07": [67.45, 25.00],
        "marker_08": [67.45, 36.20],
        "marker_09": [67.45, 47.45],
        "marker_10": [54.45, 47.45],
        "marker_11": [41.55, 47.45],
        "marker_12": [28.70, 47.45],
        "marker_13": [15.80, 47.45],
        "marker_14": [2.950, 47.45],
        "marker_15": [2.950, 36.75],
        "marker_16": [2.950, 25.55],
        "marker_17": [2.950, 14.25],
    }
    Markers = namedtuple("Markers", ["positions", "H", "W", "center", "size"])
    return Markers(marker_to_position, 50.4, 70.4, [35.2, 25.2], 5.9)


def get_intrinsics():
    return jp.array(
        [
            [611.10888672, 0.0, 315.51083374],
            [0.0, 610.02844238, 237.73669434],
            [0, 0, 1],
        ]
    )


def get_y_FOV():
    return float(2.0 * jp.arctan(480.0 / (2 * 610.02844238)))


def arg_to_marker_name(arg):
    return "marker_%02d" % (arg)


def get_num_markers(markers):
    return len(markers.positions)


def marker_name_to_arg(marker_name):
    return int(marker_name[-2:])


def is_valid_name(num_markers, marker_name):
    return marker_name_to_arg(marker_name) < num_markers


def filter_by_label(names, label):
    return list(filter(lambda name: label in name, names))


def filter_by_valid(names, num_markers):
    return list(filter(partial(is_valid_name, num_markers), names))


def cm_to_m(x_in_cm):
    return 0.01 * x_in_cm


def get_center_to_marker(markers, marker_arg):
    x, y = markers.positions[arg_to_marker_name(marker_arg)]
    x_center, y_center = markers.center
    x_to_center = x_center - x
    y_to_center = y_center - y
    # TODO why is y negative?
    center_to_marker_position = jp.array([x_to_center, -y_to_center, 0.0])
    center_to_marker_position = cm_to_m(center_to_marker_position)
    return paz.SE3.to_affine_matrix(jp.eye(3), center_to_marker_position)


def pose_to_matrix(pose):
    position = pose[0][:3]
    rotation = paz.quaternion.to_matrix(pose[0][3:])
    return paz.SE3.to_affine_matrix(rotation, position)


def build_centers_to_camera(label, markers, marker_label="ar_marker"):
    centers_to_camera = []
    num_markers = get_num_markers(markers)
    marker_names = filter_by_label(label.keys(), marker_label)
    marker_names = filter_by_valid(marker_names, num_markers)
    marker_poses = [pose_to_matrix(label[name]) for name in marker_names]
    for marker_name, marker_to_camera in zip(marker_names, marker_poses):
        marker_arg = marker_name_to_arg(marker_name)
        center_to_marker = get_center_to_marker(markers, marker_arg)
        centers_to_camera.append(marker_to_camera @ center_to_marker)
    return centers_to_camera


def load_mask(filepath):
    mask = paz.image.load(filepath, paz.image.GRAY)
    return mask.astype(bool)


def load_depth(filepath):
    depth = paz.image.load(filepath, paz.image.DEPTH)
    depth = depth / 1_000.0
    return jp.array(depth)


def load_label(path, markers):
    label = loadmat(path)
    return build_centers_to_camera(label, markers)


def load_wildcard(wildcard):
    return sorted(glob(wildcard))


def load_images(concept_path):
    image_paths = load_wildcard(os.path.join(concept_path, "*-color.jpg"))
    return [paz.image.load(path) for path in image_paths]


def load_depths(concept_path):
    depth_paths = load_wildcard(os.path.join(concept_path, "*-depth.png"))
    return [load_depth(path) for path in depth_paths]


def load_labels(concept_path, markers):
    label_paths = load_wildcard(os.path.join(concept_path, "*-meta.mat"))
    return [load_label(path, markers) for path in label_paths]


def load_masks(concept_path):
    mask_paths = load_wildcard(os.path.join(concept_path, "*-binary.png"))
    return [load_mask(path) for path in mask_paths]


def masks_to_boxes(masks):
    masks = jp.array(masks).astype("float32")
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    vectorized_to_box = jax.vmap(paz.mask.to_box, in_axes=(0, None))
    boxes = vectorized_to_box(masks, 1.0)
    boxes = paz.boxes.xyxy_to_xywh(boxes)
    return jp.expand_dims(jp.array(boxes), axis=1)


def load(concept_arg):
    root_path = download()
    concept_path = load_wildcard(os.path.join(root_path, "*"))[concept_arg]
    labels = load_labels(concept_path, get_markers())
    images = load_images(concept_path)
    depths = load_depths(concept_path)
    depth_to_points = paz.lock(paz.pointcloud.from_depth, get_intrinsics())
    points = [depth_to_points(depth) for depth in depths]
    masks = load_masks(concept_path)
    boxes = masks_to_boxes(masks)
    masks = jp.expand_dims(jp.array(masks), axis=1)  # FEWSOL has one mask
    iterator = zip(images, depths, masks, boxes, points, labels)
    return [Shot(*shot) for shot in iterator]
