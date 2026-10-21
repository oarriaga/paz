import numpy as np

from paz.models.tracking.deep_sort import kalman
from paz.models.tracking.deep_sort.tracker import TENTATIVE
from paz.models.tracking.deep_sort.tracker import CONFIRMED
from paz.models.tracking.deep_sort.tracker import Track
from paz.models.tracking.deep_sort.tracker import Detection
from paz.models.tracking.deep_sort.tracker import Config
from paz.models.tracking.deep_sort.tracker import build_tracker
from paz.models.tracking.deep_sort.tracker import update
from paz.models.tracking.deep_sort.tracker import select_valid_detections
from paz.models.tracking.deep_sort.tracker import match
from paz.models.tracking.deep_sort.tracker import predict_tracks
from paz.models.tracking.deep_sort.tracker import compute_box2D
from paz.models.tracking.deep_sort.tracker import build_appearance_costs

FISH = [1.0, 0.0, 0.0, 0.0]
CRAB = [0.0, 1.0, 0.0, 0.0]
BOX = [10.0, 10.0, 20.0, 40.0]
FAR_BOX = [400.0, 400.0, 20.0, 40.0]


def build_detection(box2D, feature, class_name):
    box2D = np.array(box2D, np.float32)
    feature = np.array(feature, np.float32)
    return Detection(box2D, 0.9, class_name, feature)


def build_fish(box2D):
    return build_detection(box2D, FISH, "fish")


def build_track(track_id, state, box2D, time_since_update):
    measurement = kalman.to_measurement(np.array(box2D, np.float32))
    args = (track_id, state, kalman.initiate(measurement), 3, 3)
    return Track(*args, time_since_update, [], "fish", None)


def build_features(num_features):
    features = []
    for feature_arg in range(num_features):
        feature = np.array([1.0, 0.01 * feature_arg, 0.0, 0.0], np.float32)
        features.append(feature / np.linalg.norm(feature))
    return features


def build_feature_frames(features):
    frames = []
    for feature in features:
        frames.append([build_detection(BOX, feature, "fish")])
    return frames


def run_frames(config, frames):
    tracker = build_tracker()
    for detections in frames:
        tracker = update(config, tracker, detections)
    return tracker


def build_hits_then_misses(num_hits, num_misses):
    return [[build_fish(BOX)]] * num_hits + [[]] * num_misses


def count_samples(tracker):
    counts = {}
    for track_id, samples in tracker.samples.items():
        counts[track_id] = len(samples)
    return counts


def list_ids(tracker):
    return [track.id for track in tracker.tracks]


def test_config_defaults_are_the_reference_defaults():
    fields = "max_age min_hits max_IOU_distance max_cosine_distance"
    assert Config._fields == tuple(fields.split())
    assert tuple(Config()) == (30, 3, 0.7, 0.2)


def test_build_tracker_starts_empty():
    tracker = build_tracker()
    assert len(tracker.tracks) == 0
    assert dict(tracker.samples) == {}
    assert tracker.next_id == 1


def test_empty_frame_keeps_tracker_empty():
    tracker = update(Config(), build_tracker(), [])
    assert len(tracker.tracks) == 0
    assert dict(tracker.samples) == {}
    assert tracker.next_id == 1


def test_new_track_fields():
    detection = build_fish(BOX)
    tracker = run_frames(Config(), [[detection]])
    (track,) = tracker.tracks
    assert (track.id, track.state, track.hits) == ("1", TENTATIVE, 1)
    assert track.age == 1
    assert track.time_since_update == 0
    assert (track.class_name, track.score) == ("fish", 0.9)
    assert len(track.features) == 1
    assert np.array_equal(track.features[0], detection.feature)
    assert tracker.next_id == 2


def test_new_track_mean_is_center_aspect_height():
    (track,) = run_frames(Config(), [[build_fish(BOX)]]).tracks
    expected = [20.0, 30.0, 0.5, 40.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(track.kalman.mean, expected, rtol=0, atol=1e-6)


def test_new_track_box_is_detection_box():
    (track,) = run_frames(Config(), [[build_fish(BOX)]]).tracks
    assert np.allclose(compute_box2D(track), BOX, rtol=0, atol=1e-4)


def test_matched_track_runs_the_kalman_update():
    moved_box2D = [14.0, 10.0, 20.0, 40.0]
    frames = [[build_fish(BOX)], [build_fish(moved_box2D)]]
    (track,) = run_frames(Config(), frames).tracks
    initial = kalman.initiate(kalman.to_measurement(np.array(BOX, np.float32)))
    measurement = kalman.to_measurement(np.array(moved_box2D, np.float32))
    expected = kalman.update(kalman.predict(initial), measurement)
    covariance = track.kalman.covariance
    assert np.allclose(track.kalman.mean, expected.mean, rtol=0, atol=1e-9)
    assert np.allclose(covariance, expected.covariance, rtol=0, atol=1e-9)
    # gain on x: predicted P00 26.25 over P00 + R00 = 30.25
    assert np.isclose(track.kalman.mean[0], 20.0 + 4.0 * 26.25 / 30.25)
    assert 10.0 < compute_box2D(track)[0] < 14.0


def test_track_is_confirmed_at_min_hits():
    tracker = run_frames(Config(), build_hits_then_misses(2, 0))
    (track,) = tracker.tracks
    assert (track.state, track.hits) == (TENTATIVE, 2)
    (track,) = update(Config(), tracker, [build_fish(BOX)]).tracks
    assert (track.state, track.hits) == (CONFIRMED, 3)


def test_min_hits_one_is_tentative_at_birth_and_confirms_on_second_hit():
    config = Config(min_hits=1)
    tracker = run_frames(config, build_hits_then_misses(1, 0))
    assert [track.state for track in tracker.tracks] == [TENTATIVE]
    tracker = update(config, tracker, [build_fish(BOX)])
    assert [track.state for track in tracker.tracks] == [CONFIRMED]


def test_tentative_track_is_deleted_on_first_miss():
    tracker = run_frames(Config(), build_hits_then_misses(2, 1))
    assert len(tracker.tracks) == 0


def test_confirmed_track_survives_max_age_misses():
    tracker = run_frames(Config(max_age=5), build_hits_then_misses(3, 5))
    (track,) = tracker.tracks
    assert (track.state, track.time_since_update) == (CONFIRMED, 5)


def test_confirmed_track_is_deleted_after_max_age_plus_one_misses():
    tracker = run_frames(Config(max_age=5), build_hits_then_misses(3, 6))
    assert len(tracker.tracks) == 0
    assert dict(tracker.samples) == {}


def test_track_returning_at_max_age_is_matched():
    frames = build_hits_then_misses(3, 4) + [[build_fish(BOX)]]
    (track,) = run_frames(Config(max_age=5), frames).tracks
    assert (track.id, track.hits, track.time_since_update) == ("1", 4, 0)


def test_track_returning_after_max_age_gets_new_id():
    frames = build_hits_then_misses(3, 5) + [[build_fish(BOX)]]
    tracker = run_frames(Config(max_age=5), frames)
    assert list_ids(tracker) == ["2"]
    assert tracker.tracks[0].state == TENTATIVE


def test_age_and_time_since_update_count_frames():
    (track,) = run_frames(Config(), build_hits_then_misses(3, 2)).tracks
    assert (track.age, track.time_since_update, track.hits) == (5, 2, 3)


def test_ids_are_never_reused():
    frames = [[build_fish(BOX)], [], [build_fish(BOX)]]
    tracker = run_frames(Config(), frames)
    assert list_ids(tracker) == ["2"]
    assert tracker.next_id == 3


def test_new_tracks_follow_detection_order():
    lefts = [300.0, 10.0, 150.0]
    features = [FISH, CRAB, [0.0, 0.0, 1.0, 0.0]]
    detections = []
    for left, feature in zip(lefts, features):
        box2D = [left, 10.0, 20.0, 40.0]
        detections.append(build_detection(box2D, feature, "fish"))
    tracker = run_frames(Config(), [detections])
    assert list_ids(tracker) == ["1", "2", "3"]
    boxes2D = [compute_box2D(track) for track in tracker.tracks]
    assert np.allclose(np.array(boxes2D)[:, 0], lefts, rtol=0, atol=1e-4)


def test_new_ids_follow_unmatched_detection_order():
    detections = []
    for left in [400.0, 500.0, 600.0]:
        box2D = [left, 400.0, 20.0, 40.0]
        detections.append(build_detection(box2D, CRAB, "fish"))
    frames = build_hits_then_misses(3, 0) + [detections]
    tracker = run_frames(Config(), frames)
    assert list_ids(tracker) == ["1", "2", "3", "4"]
    lefts = [compute_box2D(track)[0] for track in tracker.tracks[1:]]
    assert np.allclose(lefts, [600.0, 400.0, 500.0], rtol=0, atol=1e-4)


def test_gallery_counts_confirmed_tracks_with_duplicated_last_feature():
    frames = build_feature_frames(build_features(4)) + [[]]
    tracker = build_tracker()
    counts = []
    for frame in frames:
        tracker = update(Config(), tracker, frame)
        counts.append(count_samples(tracker))
    assert counts == [{}, {}, {"1": 3}, {"1": 5}, {"1": 6}]


def test_confirmed_track_keeps_only_last_feature():
    features = build_features(4)
    frames = build_feature_frames(features)
    tracker = run_frames(Config(), frames[:3])
    assert len(tracker.tracks[0].features) == 1
    assert np.array_equal(tracker.tracks[0].features[0], features[2])
    tracker = run_frames(Config(), frames + [[]])
    assert len(tracker.tracks[0].features) == 1
    assert np.array_equal(tracker.tracks[0].features[0], features[3])


def test_tentative_track_keeps_all_features():
    features = build_features(2)
    (track,) = run_frames(Config(), build_feature_frames(features)).tracks
    assert len(track.features) == 2
    assert np.array_equal(track.features[0], features[0])


def test_class_switches_to_last_matched_class():
    frames = build_hits_then_misses(3, 0)
    frames = frames + [[build_detection(BOX, FISH, "crab")]]
    (track,) = run_frames(Config(), frames).tracks
    assert (track.class_name, track.score) == ("crab", 0.9)


def test_missed_track_clears_score_and_keeps_class():
    (track,) = run_frames(Config(), build_hits_then_misses(3, 1)).tracks
    assert (track.class_name, track.score) == ("fish", None)


def test_box_outside_frame_is_kept():
    box2D = [-50.0, -30.0, 20.0, 40.0]
    tracker = run_frames(Config(), [[build_fish(box2D)]] * 3)
    (track,) = tracker.tracks
    assert np.allclose(compute_box2D(track), box2D, rtol=0, atol=1e-4)


def test_appearance_change_is_recovered_by_IOU():
    frames = build_hits_then_misses(3, 0)
    frames = frames + [[build_detection(BOX, CRAB, "fish")]]
    tracker = run_frames(Config(), frames)
    assert list_ids(tracker) == ["1"]
    assert tracker.tracks[0].hits == 4


def test_appearance_change_after_a_miss_starts_new_track():
    frames = build_hits_then_misses(3, 1)
    frames = frames + [[build_detection(BOX, CRAB, "fish")]]
    tracker = run_frames(Config(), frames)
    assert list_ids(tracker) == ["1", "2"]
    assert tracker.tracks[0].time_since_update == 2


def test_far_detection_is_gated_out():
    frames = build_hits_then_misses(3, 0) + [[build_fish(FAR_BOX)]]
    tracker = run_frames(Config(), frames)
    assert list_ids(tracker) == ["1", "2"]
    assert tracker.tracks[0].time_since_update == 1


def test_select_valid_detections_drops_empty_and_nan_sizes():
    sizes = [(20.0, 40.0), (0.0, 40.0), (20.0, -1.0), (np.nan, 40.0)]
    sizes = sizes + [(20.0, np.nan), (1e-3, 1e-3)]
    detections = []
    for size_arg, (W, H) in enumerate(sizes):
        box2D = np.array([5.0, 5.0, W, H], np.float32)
        feature = np.full(4, size_arg, np.float32)
        detections.append(Detection(box2D, 0.1 * size_arg, "fish", feature))
    valid = select_valid_detections(detections)
    assert [detection.score for detection in valid] == [0.0, 0.5]
    assert np.array_equal(valid[1].feature, detections[5].feature)


def test_predict_tracks_ages_tracks_and_clears_score():
    (track,) = run_frames(Config(), [[build_fish(BOX)]]).tracks
    (predicted,) = predict_tracks([track])
    assert (predicted.id, predicted.hits) == ("1", 1)
    assert predicted.state == TENTATIVE
    assert (predicted.age, predicted.time_since_update) == (2, 1)
    assert (predicted.class_name, predicted.score) == ("fish", None)
    expected = kalman.predict(track.kalman).covariance
    assert np.allclose(predicted.kalman.covariance, expected, rtol=1e-12)


def test_match_appearance_within_gate():
    tracks = [build_track("1", CONFIRMED, BOX, 1)]
    detections = [build_fish(BOX)]
    samples = {"1": [np.array(FISH, np.float32)]}
    result = match(Config(), tracks, samples, detections)
    assert result == ([(0, 0)], [], [])


def test_match_gates_out_far_detection():
    tracks = [build_track("1", CONFIRMED, BOX, 1)]
    detections = [build_fish(FAR_BOX)]
    samples = {"1": [np.array(FISH, np.float32)]}
    result = match(Config(), tracks, samples, detections)
    assert result == ([], [0], [0])


def test_match_IOU_candidates_are_unconfirmed_then_recent_tracks():
    boxes2D = [[100.0, 100.0, 20.0, 40.0], [300.0, 100.0, 20.0, 40.0]]
    boxes2D.append([500.0, 100.0, 20.0, 40.0])
    tracks = [build_track("1", CONFIRMED, boxes2D[0], 1)]
    tracks.append(build_track("2", TENTATIVE, boxes2D[1], 1))
    tracks.append(build_track("3", CONFIRMED, boxes2D[2], 2))
    samples = {"1": [np.array(FISH, np.float32)]}
    samples["3"] = [np.array(FISH, np.float32)]
    detections = []
    for box2D in [boxes2D[1], boxes2D[0], boxes2D[2]]:
        detections.append(build_detection(box2D, CRAB, "fish"))
    result = match(Config(), tracks, samples, detections)
    assert result == ([(1, 0), (0, 1)], [2], [2])


def test_match_without_detections_lists_tracks_in_set_order():
    tracks = [build_track("1", CONFIRMED, BOX, 1)]
    tracks.append(build_track("2", TENTATIVE, FAR_BOX, 1))
    samples = {"1": [np.array(FISH, np.float32)]}
    result = match(Config(), tracks, samples, [])
    assert result == ([], [0, 1], [])


def test_match_lists_unmatched_tracks_in_set_order():
    tracks = []
    for track_arg in range(10):
        box2D = [100.0 * track_arg, 100.0, 20.0, 40.0]
        tracks.append(build_track(str(track_arg + 1), CONFIRMED, box2D, 1))
    detections = []
    for track_arg in [0, 1, 2, 4, 5, 6, 7, 8]:
        box2D = [100.0 * track_arg, 100.0, 20.0, 40.0]
        detections.append(build_fish(box2D))
    feature = np.array(FISH, np.float32)
    samples = {track.id: [feature] for track in tracks}
    matches, unmatched_tracks, _ = match(Config(), tracks, samples, detections)
    assert [track_arg for track_arg, _ in matches] == [0, 1, 2, 4, 5, 6, 7, 8]
    assert unmatched_tracks == [9, 3]


def test_update_filters_invalid_detections_itself():
    flat = build_fish([10.0, 10.0, 20.0, 0.0])
    tracker = update(Config(), build_tracker(), [build_fish(FAR_BOX), flat])
    assert list_ids(tracker) == ["1"]


def test_update_after_an_invalid_detection_does_not_raise():
    flat = build_fish([10.0, 10.0, 20.0, 0.0])
    frames = [[build_fish(FAR_BOX), flat], [build_fish(FAR_BOX)]]
    tracker = run_frames(Config(), frames)
    assert [(track.id, track.hits) for track in tracker.tracks] == [("1", 2)]


def test_select_valid_detections_drops_non_finite_boxes():
    boxes2D = [[np.nan, 10.0, 20.0, 30.0], [10.0, np.inf, 20.0, 30.0]]
    boxes2D = boxes2D + [[10.0, 10.0, np.inf, 30.0], BOX, FAR_BOX]
    detections = [build_fish(box2D) for box2D in boxes2D]
    valid = select_valid_detections(detections)
    assert [detection.box2D.tolist() for detection in valid] == [BOX, FAR_BOX]


def test_update_drops_non_finite_boxes():
    broken = build_fish([np.nan, 10.0, 20.0, 30.0])
    tracker = update(Config(), build_tracker(), [broken, build_fish(BOX)])
    assert list_ids(tracker) == ["1"]
    assert np.allclose(compute_box2D(tracker.tracks[0]), BOX, atol=1e-4)


def test_update_accepts_list_box2D():
    box2D = [10.0, 10.0, 20.0, 30.0]
    feature = np.array(FISH, np.float32)
    detection = Detection(box2D, 0.9, "fish", feature)
    tracker = run_frames(Config(), [[detection], [detection]])
    expected = run_frames(Config(), [[build_fish(box2D)]] * 2)
    boxes2D = [compute_box2D(track) for track in tracker.tracks]
    expected_boxes2D = [compute_box2D(track) for track in expected.tracks]
    assert np.allclose(boxes2D, expected_boxes2D, rtol=0, atol=1e-6)
    assert [track.hits for track in tracker.tracks] == [2]


def test_match_normalises_detection_features():
    tracks = [build_track("1", CONFIRMED, BOX, 2)]
    samples = {"1": [np.array(FISH, np.float32)]}
    unit = build_detection(BOX, [0.6, 0.8, 0.0, 0.0], "fish")
    scaled = build_detection(BOX, [4.2, 5.6, 0.0, 0.0], "fish")
    unit_result = match(Config(), tracks, samples, [unit])
    assert match(Config(), tracks, samples, [scaled]) == unit_result
    assert unit_result == ([], [0], [0])


def test_appearance_costs_ignore_detection_feature_scale():
    tracks = [build_track("1", CONFIRMED, BOX, 1)]
    samples = {"1": [np.array(FISH, np.float32)]}
    scaled = build_detection(BOX, [4.2, 5.6, 0.0, 0.0], "fish")
    compute_costs = build_appearance_costs(tracks, samples, [scaled])
    costs = compute_costs([0], [0])
    assert np.allclose(costs, [[0.4]], rtol=0, atol=1e-6)
