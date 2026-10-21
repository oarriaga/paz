"""Opt-in parity check against deep-sort-realtime 1.3.2 used as an oracle.

Skipped unless ``PAZ_DEEP_SORT_PARITY=1``; with it set, a missing oracle is
an import error, never a skip. Both trackers get the same pre-filtered
detections and embeddings. Every fixture keeps each gate, cosine and IOU cost
more than 1e-4 away from its threshold, so float noise cannot flip a
decision; a frame that breaks this fails the test.

    PAZ_DEEP_SORT_PARITY=1 python -m pytest paz/models/tracking -rs
"""
import copy
import os
from collections import namedtuple
from functools import partial

import numpy as np
import pytest

from paz.models.tracking.deep_sort.tracker import Config
from paz.models.tracking.deep_sort.tracker import Detection
from paz.models.tracking.deep_sort.tracker import build_tracker
from paz.models.tracking.deep_sort.tracker import update
from paz.models.tracking.deep_sort.tracker import select_valid_detections
from paz.models.tracking.deep_sort.tracker import match
from paz.models.tracking.deep_sort.tracker import predict_tracks
from paz.models.tracking.deep_sort.tracker import compute_box2D

ENABLED = os.environ.get("PAZ_DEEP_SORT_PARITY") == "1"
REASON = "set PAZ_DEEP_SORT_PARITY=1 to compare with deep-sort-realtime"
pytestmark = pytest.mark.skipif(not ENABLED, reason=REASON)

if ENABLED:
    from deep_sort_realtime.deep_sort import iou_matching
    from deep_sort_realtime.deep_sort.kalman_filter import chi2inv95
    from deep_sort_realtime.deepsort_tracker import DeepSort

BOX = [100.0, 100.0, 40.0, 80.0]
OTHER_BOX = [300.0, 200.0, 40.0, 80.0]
RIGHTWARD = [1.0, 0.0, 0.0, 0.0]
LEFTWARD = [-1.0, 0.0, 0.0, 0.0]

MOTION = "start stop box2D velocity feature class_name hidden"
Motion = namedtuple("Motion", MOTION)
Noise = namedtuple("Noise", "position feature")
RECORD_FIELDS = "id state hits age time_since_update class_name score"
Record = namedtuple("Record", RECORD_FIELDS + " box2D mean covariance")

NOISE = Noise(1.0, (0.25, 0.65))


def build_oracle():
    kwargs = dict(max_age=30, n_init=3, max_iou_distance=0.7)
    kwargs.update(max_cosine_distance=0.2, nn_budget=None, embedder=None)
    return DeepSort(**kwargs)


def run_parity(frames):
    oracle, tracker = build_oracle(), build_tracker()
    for detections in frames:
        tracker = step(oracle, tracker, detections)
    return tracker


def step(oracle, tracker, detections):
    selected = select_valid_detections(detections)
    oracle_inputs = to_oracle_inputs(filter_like_oracle(detections))
    assert_same_selection(selected, oracle_inputs)
    probe = build_probe(oracle)
    oracle_detections = oracle.create_detections(*oracle_inputs)
    assert compute_margin(probe, oracle_detections) > 1e-4
    assert_same_match(tracker, probe, selected, oracle_detections)
    oracle.update_tracks(*oracle_inputs)
    tracker = update(Config(), tracker, selected)
    assert_same_state(tracker, record_oracle(oracle))
    return tracker


def filter_like_oracle(detections):
    # deepsort_tracker.py:196 filters sizes but pairs the unfiltered embeds
    # at :254, so the oracle is fed detections filtered here instead.
    valid = []
    for detection in detections:
        if detection.box2D[2] > 0 and detection.box2D[3] > 0:
            valid.append(detection)
    return valid


def to_oracle_inputs(detections):
    raw_detections, embeds = [], []
    for detection in detections:
        box2D = detection.box2D.tolist()
        raw_detections.append((box2D, detection.score, detection.class_name))
        embeds.append(detection.feature)
    return raw_detections, embeds


def build_probe(oracle):
    probe = copy.copy(oracle.tracker)
    probe.tracks = copy.deepcopy(oracle.tracker.tracks)
    probe.predict()
    return probe


def compute_margin(probe, detections):
    if len(detections) == 0:
        margin = np.inf
    else:
        margin = compute_detection_margin(probe, detections)
    return margin


def compute_detection_margin(probe, detections):
    margins = [np.inf]
    for track in probe.tracks:
        margins.append(compute_gate_margin(probe, track, detections))
        margins.append(compute_cosine_margin(probe, track, detections))
        margins.append(compute_IOU_margin(probe, track, detections))
    return min(margins)


def compute_gate_margin(probe, track, detections):
    measurements = [detection.to_xyah() for detection in detections]
    args = (track.mean, track.covariance, np.asarray(measurements))
    distances = probe.kf.gating_distance(*args)
    return np.min(np.abs(distances - chi2inv95[4]))


def compute_cosine_margin(probe, track, detections):
    if track.is_confirmed():
        features = np.array([detection.feature for detection in detections])
        costs = probe.metric.distance(features, [track.track_id])
        margin = np.min(np.abs(costs - probe.metric.matching_threshold))
    else:
        margin = np.inf
    return margin


def compute_IOU_margin(probe, track, detections):
    if track.time_since_update == 1:
        detection_args = list(range(len(detections)))
        costs = iou_matching.iou_cost([track], detections, [0], detection_args)
        margin = np.min(np.abs(costs - probe.max_iou_distance))
    else:
        margin = np.inf
    return margin


def assert_same_selection(selected, oracle_inputs):
    assert len(selected) == len(oracle_inputs[0])
    for detection, raw_detection, embed in zip(selected, *oracle_inputs):
        assert detection.box2D.tolist() == raw_detection[0]
        assert detection.score == raw_detection[1]
        assert np.array_equal(detection.feature, embed)


def assert_same_match(tracker, probe, selected, oracle_detections):
    predicted = predict_tracks(tracker.tracks)
    ours = match(Config(), predicted, tracker.samples, selected)
    assert to_lists(ours) == to_lists(probe._match(oracle_detections))


def to_lists(result):
    matches, unmatched_tracks, unmatched_detections = result
    pairs = []
    for track_arg, detection_arg in matches:
        pairs.append((int(track_arg), int(detection_arg)))
    track_args = [int(track_arg) for track_arg in unmatched_tracks]
    return pairs, track_args, [int(arg) for arg in unmatched_detections]


def record_oracle(oracle):
    records = [record_track(track) for track in oracle.tracker.tracks]
    return records, count_samples(oracle.tracker.metric.samples)


def record_track(track):
    fields = (track.track_id, track.state, track.hits, track.age)
    fields = fields + (track.time_since_update, track.det_class)
    arrays = (track.to_ltwh(), track.mean, track.covariance)
    copies = [np.array(array, copy=True) for array in arrays]
    return Record(*fields, track.det_conf, *copies)


def count_samples(samples):
    counts = []
    for track_id, features in samples.items():
        counts.append((track_id, len(features)))
    return counts


def assert_same_state(tracker, oracle_state):
    records, sample_counts = oracle_state
    ids = [record.id for record in records]
    assert [track.id for track in tracker.tracks] == ids
    for track, record in zip(tracker.tracks, records):
        assert_same_track(track, record)
    assert count_samples(tracker.samples) == sample_counts


def assert_same_track(track, record):
    fields = (track.id, track.state, track.hits, track.age)
    fields = fields + (track.time_since_update, track.class_name, track.score)
    assert fields == tuple(record[:7])
    box2D = compute_box2D(track)
    assert np.allclose(box2D, record.box2D, rtol=0, atol=1e-4)
    assert np.allclose(track.kalman.mean, record.mean, rtol=0, atol=1e-4)
    covariance = track.kalman.covariance
    assert np.allclose(covariance, record.covariance, rtol=1e-5, atol=1e-12)


def build_frames(motions, num_frames, noise, rng):
    frames = []
    for frame_arg in range(num_frames):
        frames.append(build_frame(motions, frame_arg, noise, rng))
    return frames


def build_frame(motions, frame_arg, noise, rng):
    detections = []
    for motion in motions:
        if is_visible(motion, frame_arg):
            detections.append(sample_detection(motion, frame_arg, noise, rng))
    return detections


def is_visible(motion, frame_arg):
    is_alive = motion.start <= frame_arg < motion.stop
    return is_alive and frame_arg not in motion.hidden


def sample_detection(motion, frame_arg, noise, rng):
    box2D = motion.box2D + (frame_arg - motion.start) * motion.velocity
    box2D = box2D + noise.position * rng.standard_normal(4)
    feature = perturb(motion.feature, noise.feature, rng)
    score = float(rng.uniform(0.3, 1.0))
    args = (box2D.astype(np.float32), score, motion.class_name, feature)
    return Detection(*args)


def perturb(feature, scale_range, rng):
    scale = rng.uniform(*scale_range)
    return (feature + scale * sample_feature(rng)).astype(np.float32)


def sample_feature(rng):
    feature = rng.standard_normal(16)
    return feature / np.linalg.norm(feature)


def build_motion(rng, frames, box2D, velocity, hidden):
    start, stop = frames
    box2D, velocity = np.array(box2D), np.array(velocity)
    class_name = str(rng.choice(["fish", "crab", "lobster"]))
    args = (start, stop, box2D, velocity, sample_feature(rng), class_name)
    return Motion(*args, frozenset(hidden))


def build_single(rng):
    velocity = [2.0, 1.0, 0.0, 0.0]
    return [build_motion(rng, (0, 60), BOX, velocity, [])], 60


def build_crossing(rng):
    left_box2D = [50.0, 200.0, 40.0, 80.0]
    right_box2D = [290.0, 200.0, 40.0, 80.0]
    velocity = 4.0 * np.array(RIGHTWARD)
    left = build_motion(rng, (0, 60), left_box2D, velocity, [])
    right = build_motion(rng, (0, 60), right_box2D, -velocity, [])
    return [left, right], 60


def build_gap(rng, num_hidden):
    hidden = range(20, 20 + num_hidden)
    num_frames = 50 + num_hidden
    velocity = [1.0, 0.5, 0.0, 0.0]
    motion = build_motion(rng, (0, num_frames), BOX, velocity, hidden)
    return [motion], num_frames


def build_cosine_rejections(rng):
    velocity = 2.0 * np.array(RIGHTWARD)
    first = build_motion(rng, (0, 60), BOX, velocity, [])
    second = build_motion(rng, (0, 60), OTHER_BOX, -velocity, [])
    return [first, second], 60


def build_gated_out(rng):
    first = build_motion(rng, (0, 20), BOX, RIGHTWARD, [])
    jumped = np.array([450.0, 300.0, 40.0, 80.0])
    second = first._replace(start=20, stop=40, box2D=jumped)
    return [first, second], 40


def build_empty_frames(rng):
    hidden = [10, 15, 16, 30, 31]
    first = build_motion(rng, (5, 40), BOX, [1.0, 1.0, 0.0, 0.0], hidden)
    second = build_motion(rng, (20, 40), OTHER_BOX, LEFTWARD, [30, 31])
    return [first, second], 46


def build_ties(rng):
    motion = build_motion(rng, (0, 15), BOX, [0.0, 0.0, 0.0, 0.0], [])
    return [motion] * 6, 15


def build_burst(rng):
    motions = []
    for motion_arg in range(10):
        hidden = [15] if motion_arg in (3, 9) else []
        corner = compute_corner(motion_arg, 5, (20.0, 20.0), (120.0, 150.0))
        motions.append(build_drifting(rng, (0, 30), corner, hidden))
    for motion_arg in range(12):
        corner = compute_corner(motion_arg, 6, (20.0, 350.0), (100.0, 100.0))
        motions.append(build_drifting(rng, (15, 30), corner, []))
    return motions, 30


def compute_corner(motion_arg, num_columns, origin, spacing):
    row, column = divmod(motion_arg, num_columns)
    return [origin[0] + spacing[0] * column, origin[1] + spacing[1] * row]


def build_drifting(rng, frames, corner, hidden):
    box2D = corner + [40.0, 80.0]
    return build_motion(rng, frames, box2D, [0.5, 0.5, 0.0, 0.0], hidden)


def build_flicker(rng):
    first = build_motion(rng, (0, 40), BOX, RIGHTWARD, range(1, 40, 2))
    second = build_motion(rng, (0, 40), OTHER_BOX, LEFTWARD, range(11, 40, 2))
    return [first, second], 40


def build_zero_sizes(rng):
    narrow_box2D = [100.0, 100.0, 10.0, 40.0]
    shrinking = [0.0, 0.0, -0.5, 0.0]
    nan_box2D = [200.0, 300.0, 40.0, np.nan]
    narrow = build_motion(rng, (0, 40), narrow_box2D, shrinking, [])
    normal = build_motion(rng, (0, 40), OTHER_BOX, LEFTWARD, [])
    broken = build_motion(rng, (0, 40), nan_box2D, RIGHTWARD, [])
    return [narrow, normal, broken], 40


def build_appearance_change(rng):
    velocity = [2.0, 1.0, 0.0, 0.0]
    first = build_motion(rng, (0, 20), BOX, velocity, [])
    box2D = first.box2D + 20.0 * first.velocity
    second = build_motion(rng, (20, 40), box2D, velocity, [])
    first = first._replace(class_name="fish")
    return [first, second._replace(class_name="crab")], 40


def build_random(rng):
    motions = []
    for _ in range(8):
        motions.append(sample_motion(rng, 300))
    return motions, 300


def sample_motion(rng, num_frames):
    start = int(rng.integers(0, num_frames - 30))
    stop = min(num_frames, start + int(rng.integers(30, 200)))
    corner = rng.uniform([0.0, 0.0], [560.0, 400.0])
    box2D = np.concatenate([corner, rng.uniform(30.0, 80.0, 2)])
    velocity = np.concatenate([rng.uniform(-2.0, 2.0, 2), [0.0, 0.0]])
    hidden = sample_hidden(rng, start, num_frames)
    return build_motion(rng, (start, stop), box2D, velocity, hidden)


def sample_hidden(rng, start, num_frames):
    hidden = np.flatnonzero(rng.random(num_frames) < 0.1).tolist()
    gap_start = start + int(rng.integers(0, 30))
    gap = range(gap_start, gap_start + int(rng.integers(0, 35)))
    return hidden + list(gap)


SCENARIOS = [
    ("single", build_single, NOISE),
    ("crossing", build_crossing, NOISE),
    ("gap_10", partial(build_gap, num_hidden=10), NOISE),
    ("cosine_rejections", build_cosine_rejections, Noise(1.0, (0.65, 0.85))),
    ("gated_out", build_gated_out, NOISE),
    ("empty_frames", build_empty_frames, NOISE),
    ("identical_ties", build_ties, Noise(0.0, (0.0, 0.0))),
    ("burst", build_burst, NOISE),
    ("flicker", build_flicker, NOISE),
    ("zero_sizes", build_zero_sizes, NOISE),
    ("appearance_change", build_appearance_change, NOISE),
]
SCENARIO_NAMES = [name for name, _, _ in SCENARIOS]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_NAMES)
def test_scenario_matches_oracle(scenario):
    _, build_scenario, noise = scenario
    rng = np.random.default_rng(0)
    motions, num_frames = build_scenario(rng)
    run_parity(build_frames(motions, num_frames, noise, rng))


@pytest.mark.parametrize("num_hidden, ids", [(29, ["1"]), (30, ["2"])])
def test_gap_around_max_age_matches_oracle(num_hidden, ids):
    rng = np.random.default_rng(0)
    motions, num_frames = build_gap(rng, num_hidden)
    tracker = run_parity(build_frames(motions, num_frames, NOISE, rng))
    assert [track.id for track in tracker.tracks] == ids


@pytest.mark.parametrize("seed", range(5))
def test_random_sequence_matches_oracle(seed):
    rng = np.random.default_rng(seed)
    motions, num_frames = build_random(rng)
    run_parity(build_frames(motions, num_frames, NOISE, rng))
