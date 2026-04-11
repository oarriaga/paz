import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from pathlib import Path

import database
import eigenfaces
import paz


def parse_args():
    parser = argparse.ArgumentParser(description="Build eigenfaces database")
    parser.add_argument("--experiments_path", default="experiments")
    parser.add_argument("--database_path", default="database")
    parser.add_argument(
        "--crop_faces", default=True, type=paz.standard.str_to_bool
    )
    return parser.parse_args()


def load_state(experiments_path):
    filepath = Path(experiments_path) / "eigenfaces_state.npz"
    if not filepath.exists():
        raise FileNotFoundError(
            "Run build_eigenfaces.py before build_database.py"
        )
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
        weights.append(
            eigenfaces.project_single(image, eigenface_state, detect_face)
        )
    return database.FaceDatabase(labels, weights)


def main():
    args = parse_args()
    eigenface_state = load_state(args.experiments_path)
    entries = load_image_entries(args.database_path)
    if len(entries) == 0:
        raise FileNotFoundError("Add images to database/images/<label>/ first")
    face_database = build_face_database(
        entries, eigenface_state, args.crop_faces
    )
    filepath = Path(args.database_path) / "database_state.npz"
    face_database.save(filepath)


if __name__ == "__main__":
    main()
