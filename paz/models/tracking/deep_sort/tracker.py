"""DeepSORT multi-object tracker: Kalman motion plus appearance matching.

Written from Wojke et al. 2017, "Simple Online and Realtime Tracking with a
Deep Association Metric" (arXiv:1703.07402). Its behaviour was checked
against deep-sort-realtime 1.3.2, used only as a test oracle.
"""
from collections import namedtuple

import numpy as np

from paz.models.tracking.deep_sort import kalman
from paz.models.tracking.deep_sort.matching import match_cascade
from paz.models.tracking.deep_sort.matching import match_minimum_cost
from paz.models.tracking.deep_sort.matching import gate_costs
from paz.models.tracking.deep_sort.matching import compute_unit_cosine_costs
from paz.models.tracking.deep_sort.matching import normalize_features
from paz.models.tracking.deep_sort.matching import compute_IOU_costs

TENTATIVE = 1
CONFIRMED = 2
DELETED = 3

TRACK_FIELDS = "id state kalman hits age time_since_update features"
Track = namedtuple("Track", TRACK_FIELDS + " class_name score")
Detection = namedtuple("Detection", "box2D score class_name feature")
Tracker = namedtuple("Tracker", "tracks samples next_id")
CONFIG_FIELDS = "max_age min_hits max_IOU_distance max_cosine_distance"
Config = namedtuple("Config", CONFIG_FIELDS, defaults=(30, 3, 0.7, 0.2))


def build_tracker():
    return Tracker([], {}, 1)


def update(config, tracker, detections):
    tracks = predict_tracks(tracker.tracks)
    args = (config, tracks, tracker.samples, detections, tracker.next_id)
    tracks, next_id = associate_detections(*args)
    tracks = remove_deleted(tracks)
    samples = fit_samples(tracker.samples, tracks)
    return Tracker(keep_last_features(tracks), samples, next_id)


def associate_detections(config, tracks, samples, detections, next_id):
    detections = select_valid_detections(detections)
    matches, _, detection_args = match(config, tracks, samples, detections)
    tracks = update_tracks(config, tracks, detections, matches)
    return start_tracks(tracks, detections, detection_args, next_id)


def select_valid_detections(detections):
    return [detection for detection in detections if is_valid(detection)]


def is_valid(detection):
    box2D = np.asarray(detection.box2D)
    is_finite = bool(np.all(np.isfinite(box2D)))
    return is_finite and box2D[2] > 0 and box2D[3] > 0


def match(config, tracks, samples, detections):
    confirmed_args, tentative_args = split_confirmed(tracks)
    args = (config, tracks, samples, detections, confirmed_args)
    cascade_matches, cascade_args, detection_args = match_appearance(*args)
    recent_args, stale_args = split_args(tracks, cascade_args, is_recent)
    args = (config, tracks, detections, tentative_args + recent_args)
    IOU_matches, IOU_args, detection_args = match_IOU(*args, detection_args)
    track_args = merge_unmatched(stale_args, IOU_args)
    return cascade_matches + IOU_matches, track_args, detection_args


def match_appearance(config, tracks, samples, detections, track_args):
    compute_costs = build_appearance_costs(tracks, samples, detections)
    detection_args = list(range(len(detections)))
    args = (config, tracks, track_args, detection_args)
    return match_cascade(compute_costs, *args)


def merge_unmatched(stale_args, IOU_args):
    # only match returns this list (update ignores it); tests and parity
    # pin its CPython set order, so it is not sorted
    return list(set(stale_args + IOU_args))


def match_IOU(config, tracks, detections, track_args, detection_args):
    compute_costs = build_IOU_costs(tracks, detections)
    args = (config.max_IOU_distance, track_args, detection_args)
    return match_minimum_cost(compute_costs, *args)


def build_appearance_costs(tracks, samples, detections):
    unit_features = normalize_features(to_features(detections))
    measurements = to_measurements(detections)

    def compute_costs(track_args, detection_args):
        track_subset = [tracks[arg] for arg in track_args]
        detection_args = np.array(detection_args)
        args = (unit_features[detection_args], measurements[detection_args])
        return compute_gated_costs(track_subset, samples, *args)

    return compute_costs


def build_IOU_costs(tracks, detections):
    def compute_costs(track_args, detection_args):
        boxes2D_A = [compute_box2D(tracks[arg]) for arg in track_args]
        boxes2D_B = [detections[arg].box2D for arg in detection_args]
        return compute_IOU_costs(np.array(boxes2D_A), np.array(boxes2D_B))

    return compute_costs


def compute_gated_costs(tracks, samples, unit_features, measurements):
    galleries = [samples[track.id] for track in tracks]
    costs = compute_unit_cosine_costs(galleries, unit_features)
    return gate_costs(costs, compute_gating_matrix(tracks, measurements))


def compute_gating_matrix(tracks, measurements):
    distances = []
    for track in tracks:
        args = (track.kalman, measurements)
        distances.append(kalman.compute_gating_distances(*args))
    return np.array(distances)


def to_features(detections):
    return np.array([detection.feature for detection in detections])


def to_measurements(detections):
    measurements = []
    for detection in detections:
        measurements.append(kalman.to_measurement(detection.box2D))
    return np.array(measurements)


def split_confirmed(tracks):
    return split_args(tracks, range(len(tracks)), is_confirmed)


def split_args(tracks, track_args, is_selected):
    selected_args = [arg for arg in track_args if is_selected(tracks[arg])]
    other_args = [arg for arg in track_args if not is_selected(tracks[arg])]
    return selected_args, other_args


def is_confirmed(track):
    return track.state == CONFIRMED


def is_recent(track):
    return track.time_since_update == 1


def update_tracks(config, tracks, detections, matches):
    matched_args = dict(matches)
    updated_tracks = []
    for track_arg, track in enumerate(tracks):
        args = (config, track, track_arg, detections, matched_args)
        updated_tracks.append(apply_match(*args))
    return updated_tracks


def apply_match(config, track, track_arg, detections, matched_args):
    if track_arg in matched_args:
        detection = detections[matched_args[track_arg]]
        updated_track = update_track(config, track, detection)
    else:
        updated_track = mark_missed(config, track)
    return updated_track


def update_track(config, track, detection):
    measurement = kalman.to_measurement(detection.box2D)
    state = kalman.update(track.kalman, measurement)
    features = track.features + [detection.feature]
    kwargs = dict(kalman=state, hits=track.hits + 1, time_since_update=0)
    kwargs.update(class_name=detection.class_name, score=detection.score)
    return update_state(config, track._replace(features=features, **kwargs))


def update_state(config, track):
    is_ready = is_confirmed(track) or track.hits >= config.min_hits
    return track._replace(state=CONFIRMED if is_ready else TENTATIVE)


def mark_missed(config, track):
    is_expired = track.time_since_update > config.max_age
    is_deleted = track.state == TENTATIVE or is_expired
    return track._replace(state=DELETED if is_deleted else track.state)


def start_tracks(tracks, detections, detection_args, next_id):
    new_tracks = []
    for track_id, detection_arg in enumerate(detection_args, next_id):
        detection = detections[detection_arg]
        # string ids as in the reference output; parity compares them
        new_tracks.append(start_track(str(track_id), detection))
    return tracks + new_tracks, next_id + len(detection_args)


def start_track(track_id, detection):
    measurement = kalman.to_measurement(detection.box2D)
    args = (track_id, TENTATIVE, kalman.initiate(measurement), 1, 1, 0)
    features = [detection.feature]
    return Track(*args, features, detection.class_name, detection.score)


def remove_deleted(tracks):
    return [track for track in tracks if track.state != DELETED]


def fit_samples(samples, tracks):
    fitted_samples = {}
    for track in tracks:
        if is_confirmed(track):
            track_samples = samples.get(track.id, []) + track.features
            fitted_samples[track.id] = track_samples
    return fitted_samples


def keep_last_features(tracks):
    return [keep_last_feature(track) for track in tracks]


def keep_last_feature(track):
    if is_confirmed(track):
        kept_track = track._replace(features=track.features[-1:])
    else:
        kept_track = track
    return kept_track


def predict_tracks(tracks):
    return [predict_track(track) for track in tracks]


def predict_track(track):
    state = kalman.predict(track.kalman)
    time_since_update = track.time_since_update + 1
    # the score belongs to the detection matched in this frame
    kwargs = dict(kalman=state, age=track.age + 1, score=None)
    return track._replace(time_since_update=time_since_update, **kwargs)


def compute_box2D(track):
    return kalman.to_box2D(track.kalman.mean)
