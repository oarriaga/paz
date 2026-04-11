from pathlib import Path

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


