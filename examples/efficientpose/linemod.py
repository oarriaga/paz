import os

import numpy as np
import yaml


LINEMOD_CAMERA_MATRIX = np.array([[572.4114, 0.0, 325.2611],
                                  [0.0, 573.57043, 242.04899],
                                  [0.0, 0.0, 1.0]], dtype="float32")

RGB_LINEMOD_MEAN = (123.675, 116.28, 103.53)


class Linemod:
    """Loads the LINEMOD_preprocessed dataset for a single object.

    Expected layout::

        data_path/data/<object_id>/{rgb,mask}/*.png
        data_path/data/<object_id>/gt.yml
        data_path/data/<object_id>/{train,test}.txt
        data_path/models/obj_<object_id>.ply
    """

    def __init__(self, path, object_id="08", split="train"):
        self.path = path
        self.object_id = object_id
        self.split = split
        self.class_names = ["background", "object"]
        self.num_classes = len(self.class_names)

    def load_data(self):
        root = os.path.join(self.path, "data", self.object_id)
        ground_truth = read_yaml(os.path.join(root, "gt.yml"))
        sample_ids = read_split(os.path.join(root, self.split + ".txt"))
        return [self.build_sample(root, ground_truth, arg)
                for arg in sample_ids]

    def build_sample(self, root, ground_truth, sample_id):
        annotation = ground_truth[sample_id][0]
        rotation = np.array(annotation["cam_R_m2c"], dtype="float32")
        translation = np.array(annotation["cam_t_m2c"], dtype="float32")
        box = np.array(annotation["obj_bb"], dtype="float32")
        return {
            "image": os.path.join(root, "rgb", "%04d.png" % sample_id),
            "mask": os.path.join(root, "mask", "%04d.png" % sample_id),
            "boxes": to_corner_box(box)[np.newaxis],
            "rotation": rotation[np.newaxis],
            "translation_raw": translation[np.newaxis],
            "class": 1,
        }


def to_corner_box(box):
    x, y, width, height = box
    return np.array([x, y, x + width, y + height, 1.0], dtype="float32")


def read_yaml(path):
    with open(path, "r") as opened_file:
        return yaml.safe_load(opened_file)


def read_split(path):
    with open(path, "r") as opened_file:
        return [int(line) for line in opened_file.read().split()]


def load_model_points(path, object_id, num_points=500):
    """Reads object vertices from its `.ply` file and resamples them to
    `num_points` rows for use in the pose loss."""
    from plyfile import PlyData
    filename = os.path.join(path, "models", "obj_%s.ply" % object_id)
    data = PlyData.read(filename)
    vertex = data["vertex"][:]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    return resample_points(points, num_points)


def read_object_diameter(path, object_id):
    info = read_yaml(os.path.join(path, "models", "models_info.yml"))
    return info[int(object_id)]["diameter"]


def resample_points(points, num_points):
    if points.shape[0] < num_points:
        padded = np.zeros((num_points, 3), dtype="float32")
        padded[: points.shape[0]] = points
        return padded
    step = max(1, (points.shape[0] // num_points))
    return points[::step][:num_points].astype("float32")
