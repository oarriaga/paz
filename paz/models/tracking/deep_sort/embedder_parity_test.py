"""Opt-in parity check of the embedder against deep-sort-realtime 1.3.2.

Skipped unless ``PAZ_DEEP_SORT_PARITY=1``; with it set, a missing oracle or
torch is an import error, never a skip. The oracle reads BGR frames and paz
reads RGB, so both get the same pixels in their own channel order. The
end-to-end run reports the smallest gap between any cosine cost and the 0.2
threshold; a gap under 1e-3 fails as inconclusive, never as a pass.

    PAZ_DEEP_SORT_PARITY=1 python -m pytest paz/models/tracking -rs -s
"""
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import functools
from collections import namedtuple

import cv2
import jax
import numpy as np
import pytest

from paz.models.tracking.deep_sort.mobilenet_v2.model import MobileNetV2
from paz.models.tracking.deep_sort.embedder import build_embedder
from paz.models.tracking.deep_sort.embedder import crop_boxes
from paz.models.tracking.deep_sort.tracker import CONFIRMED
from paz.models.tracking.deep_sort.tracker import Config
from paz.models.tracking.deep_sort.tracker import Detection
from paz.models.tracking.deep_sort.tracker import build_tracker
from paz.models.tracking.deep_sort.tracker import update
from paz.models.tracking.deep_sort.matching import normalize_features
from paz.models.tracking.deep_sort.matching import compute_unit_cosine_costs

ENABLED = os.environ.get("PAZ_DEEP_SORT_PARITY") == "1"
REASON = "set PAZ_DEEP_SORT_PARITY=1 to compare with deep-sort-realtime"
pytestmark = pytest.mark.skipif(not ENABLED, reason=REASON)

if ENABLED:
    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from deep_sort_realtime.embedder import embedder_pytorch
    from paz.models.tracking.deep_sort.mobilenet_v2 import port_weights as port

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jax.config.update("jax_default_matmul_precision", "highest")

FRAME_SIZE = (240, 320)
MIN_MARGIN = 1e-3
OBJECTS = [
    ((20.0, 20.0), (2.5, 0.5), (70, 50), "fish", (25,)),
    ((250.0, 20.0), (-1.5, 1.0), (60, 40), "crab", (12, 13, 14, 15, 25)),
    ((40.0, 160.0), (2.0, -0.3), (50, 60), "lobster", (25,)),
]
Motion = namedtuple("Motion", "corner velocity texture class_name hidden")
Frame = namedtuple("Frame", "image_RGB boxes2D scores class_names")


@functools.cache
def build_paz_embedder():
    model = MobileNetV2(weights=None)
    port.port_weights(model, embedder_pytorch.MOBILENETV2_BOTTLENECK_WTS)
    return build_embedder(model)


@functools.cache
def build_oracle_embedder():
    kwargs = dict(half=False, bgr=True, gpu=False)
    return embedder_pytorch.MobileNetv2_Embedder(**kwargs)


def build_oracle_tracker():
    kwargs = dict(max_age=30, n_init=3, max_iou_distance=0.7)
    kwargs.update(max_cosine_distance=0.2, nn_budget=None, half=False)
    kwargs.update(embedder="mobilenet", bgr=True, embedder_gpu=False)
    return DeepSort(**kwargs)


def build_texture(rng, size):
    height, width = size
    coarse = rng.integers(0, 256, (height // 8 + 1, width // 8 + 1, 3))
    coarse = coarse.astype(np.uint8)
    smooth = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    noise = rng.integers(-20, 21, (height, width, 3))
    return np.clip(smooth.astype(int) + noise, 0, 255).astype(np.uint8)


def to_BGR(image_RGB):
    return np.ascontiguousarray(image_RGB[..., ::-1])


def build_crop_boxes(rng, num_boxes):
    boxes2D = []
    for _ in range(num_boxes):
        corner = rng.uniform([-20.0, -20.0], [300.0, 220.0])
        boxes2D.append(np.concatenate([corner, rng.uniform(25.0, 120.0, 2)]))
    return np.array(boxes2D, np.float32)


def to_raw_detections(boxes2D, scores, class_names):
    raw_detections = []
    for box2D, score, class_name in zip(boxes2D, scores, class_names):
        raw_detections.append((box2D.tolist(), score, class_name))
    return raw_detections


def crop_like_oracle(image_BGR, boxes2D):
    names = ["fish"] * len(boxes2D)
    raw_detections = to_raw_detections(boxes2D, [0.9] * len(names), names)
    return DeepSort.crop_bb(image_BGR, raw_detections)[0]


def compute_relative_error(values, expected):
    return np.linalg.norm(values - expected) / np.linalg.norm(expected)


def build_recording(rng, num_frames):
    background = build_texture(rng, FRAME_SIZE)
    motions = build_motions(rng)
    frames = []
    for frame_arg in range(num_frames):
        image_RGB = render_frame(background, motions, frame_arg)
        detections = sample_detections(rng, motions, frame_arg)
        frames.append(Frame(image_RGB, *detections))
    return frames


def build_motions(rng):
    motions = []
    for corner, velocity, size, class_name, hidden in OBJECTS:
        texture = build_texture(rng, size)
        args = (np.array(corner), np.array(velocity), texture, class_name)
        motions.append(Motion(*args, frozenset(hidden)))
    return motions


def compute_corner(motion, frame_arg):
    return motion.corner + frame_arg * motion.velocity


def render_frame(background, motions, frame_arg):
    image_RGB = background.copy()
    for motion in motions:
        if frame_arg not in motion.hidden:
            corner = compute_corner(motion, frame_arg)
            left, top = np.round(corner).astype(int)
            height, width = motion.texture.shape[:2]
            image_RGB[top:top + height, left:left + width] = motion.texture
    return image_RGB


def sample_detections(rng, motions, frame_arg):
    boxes2D, scores, class_names = [], [], []
    for motion in motions:
        if frame_arg not in motion.hidden:
            size = np.array(motion.texture.shape[1::-1], np.float64)
            box2D = np.concatenate([compute_corner(motion, frame_arg), size])
            boxes2D.append(box2D + rng.normal(0.0, 0.5, 4))
            scores.append(float(rng.uniform(0.5, 1.0)))
            class_names.append(motion.class_name)
    boxes2D = np.array(boxes2D, np.float32).reshape(-1, 4)
    return boxes2D, scores, class_names


def step(oracle, tracker, frame):
    detections = embed_detections(frame)
    margin = compute_cosine_margin(tracker, detections)
    args = (frame.boxes2D, frame.scores, frame.class_names)
    raw_detections = to_raw_detections(*args)
    image_BGR = to_BGR(frame.image_RGB)
    oracle_tracks = oracle.update_tracks(raw_detections, frame=image_BGR)
    tracker = update(Config(), tracker, detections)
    assert describe_tracks(tracker.tracks) == describe_oracle(oracle_tracks)
    return tracker, margin


def embed_detections(frame):
    detections = []
    if len(frame.boxes2D) > 0:
        embed = build_paz_embedder()
        features = np.asarray(embed(frame.image_RGB, frame.boxes2D))
        args = (frame.boxes2D, frame.scores, frame.class_names, features)
        detections = [Detection(*fields) for fields in zip(*args)]
    return detections


def compute_cosine_margin(tracker, detections):
    galleries = []
    for track in tracker.tracks:
        if track.state == CONFIRMED:
            galleries.append(tracker.samples[track.id])
    has_costs = len(galleries) > 0 and len(detections) > 0
    margin = np.inf
    if has_costs:
        margin = compute_gallery_margin(galleries, detections)
    return margin


def compute_gallery_margin(galleries, detections):
    features = [detection.feature for detection in detections]
    costs = compute_unit_cosine_costs(galleries, normalize_features(features))
    return float(np.min(np.abs(costs - Config().max_cosine_distance)))


def describe_tracks(tracks):
    descriptions = []
    for track in tracks:
        fields = (track.id, track.state, track.hits, track.age)
        fields = fields + (track.time_since_update, track.class_name)
        descriptions.append(fields)
    return descriptions


def describe_oracle(tracks):
    descriptions = []
    for track in tracks:
        fields = (track.track_id, track.state, track.hits, track.age)
        fields = fields + (track.time_since_update, track.det_class)
        descriptions.append(fields)
    return descriptions


def test_crops_match_oracle():
    rng = np.random.default_rng(0)
    image_RGB = build_texture(rng, FRAME_SIZE)
    boxes2D = build_crop_boxes(rng, 20)
    crops = crop_boxes(image_RGB, boxes2D)
    oracle_crops = crop_like_oracle(to_BGR(image_RGB), boxes2D)
    assert len(crops) == len(oracle_crops) == 20
    for crop, oracle_crop in zip(crops, oracle_crops):
        assert np.array_equal(crop, oracle_crop[..., ::-1])


def test_features_match_oracle():
    rng = np.random.default_rng(1)
    image_RGB = build_texture(rng, FRAME_SIZE)
    boxes2D = build_crop_boxes(rng, 20)
    oracle_crops = crop_like_oracle(to_BGR(image_RGB), boxes2D)
    expected = np.array(build_oracle_embedder().predict(oracle_crops))
    features = np.asarray(build_paz_embedder()(image_RGB, boxes2D))
    assert features.shape == expected.shape == (20, 1280)
    assert np.allclose(features, expected, rtol=0, atol=1e-4)
    assert compute_relative_error(features, expected) <= 1e-5


def test_tracking_with_embedder_matches_oracle():
    rng = np.random.default_rng(2)
    oracle, tracker, margins = build_oracle_tracker(), build_tracker(), []
    for frame in build_recording(rng, 40):
        tracker, margin = step(oracle, tracker, frame)
        margins.append(margin)
    margin = min(margins)
    print(f"smallest cosine margin to 0.2: {margin:.6f}")
    assert np.isfinite(margin)
    assert margin >= MIN_MARGIN, f"inconclusive: cosine margin {margin:.6f}"
    assert [track.id for track in tracker.tracks] == ["1", "2", "3"]
