import os
import argparse
from pathlib import Path

import database
import eigenfaces
import paz
import numpy as np


class FaceDatabase:
    def __init__(self, labels, weights):
        self.labels = tuple(labels)
        self.weights = np.asarray(weights, dtype=np.float32)

    def query(self, weight, threshold=None):
        if len(self.labels) == 0:
            raise ValueError("database is empty")
        weight = np.asarray(weight, dtype=np.float32)
        diff = self.weights - np.expand_dims(weight, axis=0)
        distances = np.linalg.norm(diff, axis=1)
        best_arg = int(np.argmin(distances))
        best_distance = float(distances[best_arg])
        if threshold is not None and best_distance > threshold:
            return "Face not found"
        return self.labels[best_arg]

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = dict(labels=np.array(self.labels), weights=self.weights)
        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath):
        data = np.load(Path(filepath))
        labels = tuple(label.item() for label in data["labels"])
        weights = np.array(data["weights"], dtype=np.float32)
        return cls(labels, weights)


def load_state(experiments_path):
    filepath = Path(experiments_path) / "eigenfaces_state.npz"
    if not filepath.exists():
        raise FileNotFoundError("Run build_eigenfaces.py before this script")
    return eigenfaces.load(filepath)


def load_image_entries(database_path):
    image_root = Path(database_path) / "images"
    entries = []
    for image_path in sorted(image_root.glob("*/*")):
        if image_path.is_file():
            entries.append((image_path.parent.name, image_path))
    return entries


def build_face_database(entries, eigenface_state, crop_faces):
    labels, weights = [], []
    detect_face = paz.models.HaarCascadeFrontalFaceDetector(draw=None)
    detect_face = detect_face if crop_faces else None
    for label, image_path in entries:
        image = paz.image.load(image_path)
        labels.append(label)
        args = (image, eigenface_state, detect_face)
        weights.append(eigenfaces.project_single(*args))
    return database.FaceDatabase(labels, weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build eigenfaces database")
    parser.add_argument("--experiments_path", default="experiments")
    parser.add_argument("--database_path", default="database")
    parser.add_argument("--crop_faces", default=True)
    args = parser.parse_args()

    eigenface_state = load_state(args.experiments_path)
    entries = load_image_entries(args.database_path)
    if len(entries) == 0:
        raise FileNotFoundError("Add images to database/images/<label>/ first")
    database = build_face_database(entries, eigenface_state, args.crop_faces)
    filepath = Path(args.database_path) / "database_state.npz"
    database.save(filepath)
