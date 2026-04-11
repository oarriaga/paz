import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
from pathlib import Path

import cv2
import jax.numpy as jp
import numpy as np

import database
import eigenfaces
import paz


def parse_args():
    parser = argparse.ArgumentParser(description="Eigenfaces live demo")
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--H", default=480, type=int)
    parser.add_argument("--W", default=640, type=int)
    parser.add_argument("--box_scale", default=1.2, type=float)
    parser.add_argument("--distance_threshold", default=None, type=float)
    parser.add_argument("--experiments_path", default="experiments")
    parser.add_argument("--database_path", default="database")
    return parser.parse_args()


def load_state(experiments_path):
    filepath = Path(experiments_path) / "eigenfaces_state.npz"
    if not filepath.exists():
        raise FileNotFoundError("Run build_eigenfaces.py before demo.py")
    return eigenfaces.load(filepath)


def load_database(database_path):
    filepath = Path(database_path) / "database_state.npz"
    if not filepath.exists():
        raise FileNotFoundError("Run build_database.py before demo.py")
    return database.FaceDatabase.load(filepath)


def build_demo_pipeline(state, face_database, box_scale=1.2, threshold=None):
    detect_face = paz.models.HaarCascadeFrontalFaceDetector(draw=None)
    color_by_label = _build_color_by_label(face_database.labels)

    def call(image):
        boxes = _build_face_boxes(image, detect_face, box_scale)
        labels = []
        for box in boxes:
            face = paz.image.crop(image, _cast_box(box))
            weight = eigenfaces.project_single(face, state)
            label = face_database.query(weight, threshold)
            labels.append(label)
        image_with_boxes = _draw_predictions(image, boxes, labels, color_by_label)
        return (boxes, labels), image_with_boxes

    return call


def main():
    args = parse_args()
    eigenface_state = load_state(args.experiments_path)
    face_database = load_database(args.database_path)
    box_scale = args.box_scale
    threshold = args.distance_threshold
    build_args = (eigenface_state, face_database, box_scale, threshold)
    pipeline = build_demo_pipeline(*build_args)
    camera = paz.Camera(args.camera)
    player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
    player.run()


def _build_face_boxes(image, detect_face, box_scale):
    H, W = paz.image.get_size(image)
    boxes = paz.detection.get_boxes(detect_face(image))
    boxes = paz.boxes.remove_invalid(boxes)
    boxes = paz.boxes.square(boxes)
    boxes = paz.boxes.scale(boxes, box_scale, box_scale)
    boxes = paz.boxes.clip(boxes, H, W)
    return paz.boxes.remove_invalid(paz.cast(boxes, "int32"))


def _draw_predictions(image, boxes, labels, color_by_label):
    image = np.ascontiguousarray(paz.to_numpy(image))
    for box, label in zip(boxes, labels):
        color = color_by_label[label]
        image = _draw_prediction(image, box, label, color)
    return image


def _draw_prediction(image, box, label, color):
    image = paz.draw.box(image, _cast_box(box).tolist(), color, 2)
    return _draw_label(image, box, label, color)


def _draw_label(image, box, label, color):
    x_min, y_min = _cast_box(box)[:2]
    y_text = max(int(y_min) - 8, 12)
    origin = (int(x_min), y_text)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    return image


def _build_color_by_label(labels):
    unique_labels = tuple(dict.fromkeys(labels))
    colors = paz.draw.lincolor(len(unique_labels))
    color_by_label = dict(zip(unique_labels, colors))
    color_by_label["Face not found"] = paz.draw.RED
    return color_by_label


def _cast_box(box):
    return jp.array(box, dtype=jp.int32)


if __name__ == "__main__":
    main()
