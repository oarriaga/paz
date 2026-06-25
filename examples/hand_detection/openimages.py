import os
import csv

import numpy as np


HAND_CLASS = "Human hand"


class OpenImagesV6Hand:
    """Loads the `Human hand` subset of Open Images V6.

    Expected layout (FiftyOne export)::

        path/<split>/data/<image_id>.jpg
        path/<split>/metadata/classes.csv          # machine_id,human_name
        path/<split>/labels/detections.csv         # ImageID,...,LabelName,...,
                                                    #   XMin,XMax,YMin,YMax,...
    """

    def __init__(self, path, split="train"):
        self.path = path
        self.split = split
        self.class_names = ["background", "hand"]
        self.num_classes = len(self.class_names)

    def load_data(self):
        root = os.path.join(self.path, self.split)
        hand_id = self.find_hand_label(os.path.join(root, "metadata",
                                                    "classes.csv"))
        boxes = self.read_detections(os.path.join(root, "labels",
                                                  "detections.csv"), hand_id)
        images, detections = [], []
        for image_id, image_boxes in boxes.items():
            images.append(os.path.join(root, "data", image_id + ".jpg"))
            detections.append(np.array(image_boxes, dtype="float32"))
        return images, detections

    def find_hand_label(self, classes_path):
        with open(classes_path) as opened_file:
            for machine_id, human_name in csv.reader(opened_file):
                if human_name == HAND_CLASS:
                    return machine_id
        raise ValueError("Hand class not found in", classes_path)

    def read_detections(self, detections_path, hand_id):
        boxes = {}
        with open(detections_path) as opened_file:
            reader = csv.DictReader(opened_file)
            for row in reader:
                if row["LabelName"] != hand_id:
                    continue
                box = [float(row["XMin"]), float(row["YMin"]),
                       float(row["XMax"]), float(row["YMax"]), 1.0]
                boxes.setdefault(row["ImageID"], []).append(box)
        return boxes
