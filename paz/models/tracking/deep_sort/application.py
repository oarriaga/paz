"""DeepSORT video tracking application.

``DeepSORT(detect, embed, config, draw)`` returns a generator function that
follows detector boxes through the frames. Its ``detect(image)`` returns
``(boxes2D, class_names, scores)``: corner boxes ``[x_min, y_min, x_max,
y_max]`` as the paz detectors give them, but class names in place of their
integer labels. Each frame yields a ``TrackedFrame`` of the valid detections
and the live tracks; ``embed`` is skipped on frames with no valid box.
"""
import itertools
from collections import namedtuple
from functools import partial

import numpy as np

import paz
from paz.models.tracking.deep_sort.tracker import CONFIRMED
from paz.models.tracking.deep_sort.tracker import Detection
from paz.models.tracking.deep_sort.tracker import build_tracker
from paz.models.tracking.deep_sort.tracker import update
from paz.models.tracking.deep_sort.tracker import select_valid_detections
from paz.models.tracking.deep_sort.tracker import compute_box2D

TrackedFrame = namedtuple("TrackedFrame", "detections tracks")


def DeepSORT(detect, embed, config, draw):
    draw = draw_track_boxes if draw is None else draw
    call = partial(track_frames, detect, embed, config)
    return partial(track_and_draw, call, draw) if callable(draw) else call


def track_frames(detect, embed, config, frames):
    tracker = build_tracker()
    for image in frames:
        detections = detect_objects(detect, embed, image)
        tracker = update(config, tracker, detections)
        yield TrackedFrame(detections, tracker.tracks)


def track_and_draw(track, draw, frames):
    # frames may be a one-shot generator, read by both the tracker and draw
    images, inputs = itertools.tee(frames)
    for image, tracked in zip(images, track(inputs)):
        yield tracked, draw(image, tracked.tracks)


def detect_objects(detect, embed, image):
    detections = build_detections(*detect(image))
    detections = select_valid_detections(detections)
    return embed_detections(embed, image, detections)


def build_detections(boxes2D, class_names, scores):
    boxes2D = np.asarray(boxes2D, np.float32).reshape(-1, 4)
    boxes2D = np.asarray(paz.boxes.xyxy_to_xywh(boxes2D), np.float32)
    detections = []
    for box2D, class_name, score in zip(boxes2D, class_names, scores):
        args = (box2D, float(score), str(class_name), None)
        detections.append(Detection(*args))
    return detections


def embed_detections(embed, image, detections):
    if len(detections) == 0:
        embedded = detections
    else:
        embedded = attach_features(embed, image, detections)
    return embedded


def attach_features(embed, image, detections):
    boxes2D = np.array([detection.box2D for detection in detections])
    features = np.asarray(embed(image, boxes2D))
    embedded = []
    for detection, feature in zip(detections, features):
        embedded.append(detection._replace(feature=feature))
    return embedded


def draw_track_boxes(image, tracks):
    image = np.array(image, np.uint8)
    for track in tracks:
        if track.state == CONFIRMED and track.time_since_update == 0:
            image = draw_track_box(image, track, track.id)
    return image


def draw_track_box(image, track, label):
    x_min, y_min, W, H = compute_box2D(track)
    corners = np.round([x_min, y_min, x_min + W, y_min + H]).astype(int)
    image = paz.draw.box(image, corners.tolist())
    point = (int(corners[0]), int(corners[1]) - 5)
    return paz.draw.text(image, label, point)
