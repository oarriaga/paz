import os

import numpy as np
import pandas as pd


def load(path, split="train"):
    filename = {"train": "training.csv", "test": "test.csv"}[split]
    data_frame = pd.read_csv(os.path.join(path, filename))
    data_frame = data_frame.fillna(method="ffill")
    images = load_faces(data_frame)
    if split != "train":
        return images, None
    return images, load_keypoints(data_frame)


def load_faces(data_frame):
    faces = np.zeros((len(data_frame), 96, 96), "float32")
    for arg, face in enumerate(data_frame.Image):
        faces[arg] = np.array(face.split(" "), "float32").reshape(96, 96)
    return faces


def load_keypoints(data_frame):
    columns = data_frame.iloc[:, :-1].to_numpy("float32")
    return columns.reshape(len(data_frame), 15, 2)
