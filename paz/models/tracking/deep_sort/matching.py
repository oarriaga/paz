"""Cost matrices and minimum-cost assignment for the DeepSORT tracker."""
import numpy as np
import jax.numpy as jp
from scipy.optimize import linear_sum_assignment

import paz

INFINITE_COST = 1e5
CHI2_95_4D = 9.4877


def match_cascade(compute_costs, config, tracks, track_args, detection_args):
    matches, unmatched_detections = [], list(detection_args)
    for time_since_update in range(1, config.max_age + 1):
        level_args = select_by_age(tracks, track_args, time_since_update)
        args = (compute_costs, config, level_args, unmatched_detections)
        level_matches, unmatched_detections = match_level(*args)
        matches = matches + level_matches
    return matches, subtract_matched(track_args, matches), unmatched_detections


def match_level(compute_costs, config, track_args, detection_args):
    args = (config.max_cosine_distance, track_args, detection_args)
    matches, _, unmatched_detections = match_minimum_cost(compute_costs, *args)
    return matches, unmatched_detections


def select_by_age(tracks, track_args, time_since_update):
    selected_args = []
    for track_arg in track_args:
        if tracks[track_arg].time_since_update == time_since_update:
            selected_args.append(track_arg)
    return selected_args


def subtract_matched(track_args, matches):
    matched_args = {track_arg for track_arg, _ in matches}
    # CPython set order, not sorted: it fixes the order of the new track ids
    return list(set(track_args) - matched_args)


def match_minimum_cost(compute_costs, max_cost, track_args, detection_args):
    if len(track_args) == 0 or len(detection_args) == 0:
        result = ([], list(track_args), list(detection_args))
    else:
        args = (compute_costs, max_cost, track_args, detection_args)
        result = solve_assignment(*args)
    return result


def solve_assignment(compute_costs, max_cost, track_args, detection_args):
    args = (compute_costs, max_cost, track_args, detection_args)
    assignment = compute_assignment(*args)
    return split_assignment(track_args, detection_args, *assignment)


def compute_assignment(compute_costs, max_cost, track_args, detection_args):
    costs = clip_costs(compute_costs(track_args, detection_args), max_cost)
    rows, columns = linear_sum_assignment(costs)
    return rows, columns, costs[rows, columns] <= max_cost


def split_assignment(track_args, detection_args, rows, columns, is_accepted):
    args = (track_args, rows, is_accepted)
    matched_tracks, unmatched_tracks = split_assigned(*args)
    args = (detection_args, columns, is_accepted)
    matched_detections, unmatched_detections = split_assigned(*args)
    matches = list(zip(matched_tracks, matched_detections))
    return matches, unmatched_tracks, unmatched_detections


def split_assigned(candidate_args, positions, is_accepted):
    accepted = take(candidate_args, positions[is_accepted])
    rejected = take(candidate_args, positions[~is_accepted])
    unassigned = list_unassigned(candidate_args, positions)
    return accepted, unassigned + rejected


def list_unassigned(candidate_args, positions):
    unassigned = []
    for position, candidate_arg in enumerate(candidate_args):
        if position not in positions:
            unassigned.append(candidate_arg)
    return unassigned


def take(candidate_args, positions):
    return [candidate_args[position] for position in positions]


def clip_costs(costs, max_cost):
    return np.where(costs > max_cost, max_cost + 1e-5, costs)


def gate_costs(costs, distances):
    return np.where(distances > CHI2_95_4D, INFINITE_COST, costs)


def compute_unit_cosine_costs(galleries, unit_features):
    rows = []
    for samples in galleries:
        rows.append(compute_unit_distances(samples, unit_features))
    return np.asarray(np.stack(rows), np.float64)


def compute_unit_distances(samples, unit_features):
    unit_samples = normalize_features(samples)
    return np.min(1.0 - unit_samples @ unit_features.T, axis=0)


def normalize_features(features):
    # numpy: eager JAX recompiles per gallery size; 300 frames on CPU took
    # 54.4 s cold and 12.7-13.8 s warm in JAX, 8.5-8.6 s in numpy
    features = np.asarray(features, np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def compute_IOU_costs(boxes2D_A, boxes2D_B):
    boxes_A = paz.boxes.xywh_to_xyxy(jp.asarray(boxes2D_A))
    boxes_B = paz.boxes.xywh_to_xyxy(jp.asarray(boxes2D_B))
    IOUs = paz.boxes.compute_IOUs(boxes_A, boxes_B)
    return 1.0 - np.asarray(IOUs, np.float64)
