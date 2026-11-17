import numpy as np
import pytest

from paz.models.tracking.deep_sort.matching import INFINITE_COST
from paz.models.tracking.deep_sort.matching import CHI2_95_4D
from paz.models.tracking.deep_sort.matching import match_cascade
from paz.models.tracking.deep_sort.matching import match_minimum_cost
from paz.models.tracking.deep_sort.matching import clip_costs
from paz.models.tracking.deep_sort.matching import gate_costs
from paz.models.tracking.deep_sort.matching import compute_unit_cosine_costs
from paz.models.tracking.deep_sort.matching import normalize_features
from paz.models.tracking.deep_sort.matching import compute_IOU_costs
from paz.models.tracking.deep_sort.tracker import CONFIRMED
from paz.models.tracking.deep_sort.tracker import Config
from paz.models.tracking.deep_sort.tracker import Track


def compute_costs_of_raw_features(galleries, features):
    return compute_unit_cosine_costs(galleries, normalize_features(features))


def raise_if_called(track_args, detection_args):
    raise AssertionError("the cost function must not be called")


def build_lookup(costs):
    costs = np.array(costs, dtype=np.float64)

    def compute_costs(track_args, detection_args):
        return costs[np.ix_(track_args, detection_args)].copy()

    return compute_costs


def build_track(time_since_update):
    args = ("1", CONFIRMED, None, 3, 3, time_since_update, [], "fish", 0.9)
    return Track(*args)


def build_tracks(times_since_update):
    tracks = []
    for time_since_update in times_since_update:
        tracks.append(build_track(time_since_update))
    return tracks


def test_minimum_cost_without_detections_skips_costs():
    result = match_minimum_cost(raise_if_called, 0.2, [0, 1], [])
    assert result == ([], [0, 1], [])


def test_minimum_cost_without_tracks_skips_costs():
    result = match_minimum_cost(raise_if_called, 0.2, [], [0, 1])
    assert result == ([], [], [0, 1])


def test_clip_costs_sets_rejected_costs_just_above_maximum():
    costs = np.array([[0.1, 0.2, 0.3, 100.0]])
    expected = np.array([[0.1, 0.2, 0.2 + 1e-5, 0.2 + 1e-5]])
    assert np.array_equal(clip_costs(costs, 0.2), expected)


def test_clipping_changes_the_assignment():
    compute_costs = build_lookup([[0.0, 0.3], [0.1, 100.0]])
    result = match_minimum_cost(compute_costs, 0.2, [0, 1], [0, 1])
    assert result == ([(0, 0)], [1], [1])


@pytest.mark.parametrize("max_cost", [0.2, 0.7])
def test_cost_equal_to_maximum_is_accepted(max_cost):
    compute_costs = build_lookup([[max_cost]])
    result = match_minimum_cost(compute_costs, max_cost, [0], [0])
    assert result == ([(0, 0)], [], [])


@pytest.mark.parametrize("max_cost", [0.2, 0.7])
def test_cost_above_maximum_is_rejected(max_cost):
    compute_costs = build_lookup([[np.nextafter(max_cost, 1.0)]])
    result = match_minimum_cost(compute_costs, max_cost, [0], [0])
    assert result == ([], [0], [0])


def test_unmatched_detections_list_unassigned_then_rejected():
    compute_costs = build_lookup(np.full((1, 8), 0.3))
    result = match_minimum_cost(compute_costs, 0.2, [0], [5, 2, 7])
    assert result == ([], [0], [2, 7, 5])


def test_unmatched_tracks_keep_input_order():
    costs = np.full((7, 1), 0.9)
    costs[1, 0] = 0.1
    result = match_minimum_cost(build_lookup(costs), 0.2, [6, 1, 4], [0])
    assert result == ([(1, 0)], [6, 4], [])


def test_matches_carry_original_arguments_in_row_order():
    costs = np.full((8, 5), 0.9)
    costs[7, 1] = 0.05
    costs[2, 4] = 0.1
    result = match_minimum_cost(build_lookup(costs), 0.2, [7, 2], [4, 1])
    assert result == ([(7, 1), (2, 4)], [], [])


def test_cascade_without_detections_returns_set_order():
    tracks = build_tracks([1] * 10)
    result = match_cascade(raise_if_called, Config(), tracks, [3, 9], [])
    assert result == ([], [9, 3], [])


def test_cascade_prefers_recent_tracks_over_cheaper_ones():
    tracks = build_tracks([2, 1])
    compute_costs = build_lookup([[0.05], [0.15]])
    result = match_cascade(compute_costs, Config(), tracks, [0, 1], [0])
    assert result == ([(1, 0)], [0], [])


def test_cascade_passes_leftover_detections_to_next_level():
    tracks = build_tracks([1, 2])
    compute_costs = build_lookup([[0.1, 0.9], [0.05, 0.1]])
    result = match_cascade(compute_costs, Config(), tracks, [0, 1], [0, 1])
    assert result == ([(0, 0), (1, 1)], [], [])


def test_cascade_depth_reaches_max_age():
    tracks = build_tracks([3])
    compute_costs = build_lookup([[0.1]])
    result = match_cascade(compute_costs, Config(max_age=3), tracks, [0], [0])
    assert result == ([(0, 0)], [], [])


def test_cascade_depth_stops_at_max_age():
    tracks = build_tracks([3])
    compute_costs = build_lookup([[0.1]])
    result = match_cascade(compute_costs, Config(max_age=2), tracks, [0], [0])
    assert result == ([], [0], [0])


def test_cascade_uses_max_cosine_distance():
    tracks = build_tracks([1])
    compute_costs = build_lookup([[0.3]])
    config = Config(max_IOU_distance=0.25, max_cosine_distance=0.35)
    result = match_cascade(compute_costs, config, tracks, [0], [0])
    assert result == ([(0, 0)], [], [])


def test_cascade_unmatched_tracks_follow_set_difference_copy_path():
    tracks = build_tracks([1] * 40)
    track_args = list(range(12)) + list(range(32, 40))
    costs = np.full((40, 4), 0.9)
    costs[[0, 1, 2, 5], [0, 1, 2, 3]] = 0.0
    args = (build_lookup(costs), Config(), tracks, track_args, [0, 1, 2, 3])
    matches, unmatched_tracks, unmatched_detections = match_cascade(*args)
    expected = [3, 4, 6, 7, 8, 9, 10, 11] + list(range(32, 40))
    assert matches == [(0, 0), (1, 1), (2, 2), (5, 3)]
    assert unmatched_tracks == expected
    assert unmatched_detections == []


def test_gate_costs_threshold_is_inclusive():
    costs = np.array([[0.1, 0.1, 0.1]])
    distances = np.array([[7.0, CHI2_95_4D, np.nextafter(CHI2_95_4D, 10.0)]])
    expected = np.array([[0.1, 0.1, INFINITE_COST]])
    assert np.array_equal(gate_costs(costs, distances), expected)


def test_gate_costs_use_the_four_dimensional_threshold():
    costs = np.array([[0.1, 0.1]])
    distances = np.array([[9.4877, 9.4878]])
    expected = np.array([[0.1, 1e5]])
    assert np.array_equal(gate_costs(costs, distances), expected)


def test_cosine_distances_normalise_features():
    samples = np.array([[2.0, 0.0]], np.float32)
    features = np.array([[3.0, 4.0]], np.float32)
    distances = compute_costs_of_raw_features([samples], features)[0]
    assert np.allclose(distances, [0.4], rtol=0, atol=1e-6)


def test_cosine_distances_take_minimum_over_samples():
    samples = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    features = np.array([[0.6, 0.8], [1.0, 0.0]], np.float32)
    distances = compute_costs_of_raw_features([samples], features)[0]
    assert np.allclose(distances, [0.2, 0.0], rtol=0, atol=1e-6)


def test_cosine_costs_are_float32_values_in_float64_matrix():
    rng = np.random.default_rng(0)
    galleries = [rng.standard_normal((3, 8)), rng.standard_normal((1, 8))]
    galleries = [gallery.astype(np.float32) for gallery in galleries]
    features = rng.standard_normal((4, 8)).astype(np.float32)
    costs = compute_costs_of_raw_features(galleries, features)
    rounded = costs.astype(np.float32).astype(np.float64)
    assert costs.dtype == np.float64
    assert costs.shape == (2, 4)
    assert np.array_equal(costs, rounded)


def test_cosine_costs_rows_follow_galleries():
    galleries = [np.array([[1.0, 0.0]], np.float32)]
    galleries.append(np.array([[0.0, 1.0], [1.0, 1.0]], np.float32))
    features = np.array([[1.0, 0.0], [0.0, 2.0]], np.float32)
    costs = compute_costs_of_raw_features(galleries, features)
    expected = [[0.0, 1.0], [1.0 - np.sqrt(0.5), 0.0]]
    assert np.allclose(costs, expected, rtol=0, atol=1e-6)


def test_IOU_costs_use_left_top_width_height_without_plus_one():
    boxes2D_A = np.array([[0.0, 0.0, 10.0, 10.0]])
    boxes2D_B = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 10.0, 10.0]])
    boxes2D_B = np.concatenate([boxes2D_B, [[20.0, 20.0, 5.0, 5.0]]])
    costs = compute_IOU_costs(boxes2D_A, boxes2D_B)
    assert costs.dtype == np.float64
    assert np.allclose(costs, [[0.0, 2.0 / 3.0, 1.0]], rtol=0, atol=1e-6)


def test_IOU_costs_shape_is_tracks_by_detections():
    boxes2D_A = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 4.0, 5.0]])
    boxes2D_B = np.array([[0.0, 0.0, 10.0, 5.0]])
    costs = compute_IOU_costs(boxes2D_A, boxes2D_B)
    assert np.allclose(costs, [[0.5], [0.6]], rtol=0, atol=1e-6)
