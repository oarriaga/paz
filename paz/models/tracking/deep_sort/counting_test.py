"""Counting rules and track_video helpers, ending in a synthetic video run.

The parity test is opt-in: with ``PAZ_DEEP_SORT_PARITY=1`` the synthetic run
is repeated on deep-sort-realtime 1.3.2, and a missing oracle is an error.
"""
import json
import os
from collections import namedtuple

import cv2
import numpy as np
import pytest

import paz
from paz.models.tracking.deep_sort import CONFIRMED
from paz.models.tracking.deep_sort import TENTATIVE
from paz.models.tracking.deep_sort import Config
from paz.models.tracking.deep_sort import Detection
from paz.models.tracking.deep_sort import Track
from paz.models.tracking.deep_sort import compute_box2D
from paz.models.tracking.deep_sort import kalman
from paz.models.tracking.deep_sort.application import DeepSORT
from paz.models.tracking.deep_sort.counting import CountedFrame
from paz.models.tracking.deep_sort.counting import select_classes
from paz.models.tracking.deep_sort.counting import build_counting
from paz.models.tracking.deep_sort.counting import count_tracked_frames
from paz.models.tracking.deep_sort.counting import update_counting
from paz.models.tracking.deep_sort.counting import count_frame_classes
from paz.models.tracking.deep_sort.counting import update_max_counts
from paz.models.tracking.deep_sort.counting import build_frame_record
from paz.models.tracking.deep_sort.counting import to_track_records
from paz.models.tracking.deep_sort.counting import build_summary
from paz.models.tracking.deep_sort.counting import summarize_timing
from paz.models.tracking.deep_sort.track_video import build_detect
from paz.models.tracking.deep_sort.track_video import open_video
from paz.models.tracking.deep_sort.track_video import read_fps
from paz.models.tracking.deep_sort.track_video import read_frames
from paz.models.tracking.deep_sort.track_video import pair_frames
from paz.models.tracking.deep_sort.track_video import time_steps
from paz.models.tracking.deep_sort.track_video import draw_tracks
from paz.models.tracking.deep_sort.track_video import draw_counts
from paz.models.tracking.deep_sort.track_video import write_json

ENABLED = os.environ.get("PAZ_DEEP_SORT_PARITY") == "1"
REASON = "set PAZ_DEEP_SORT_PARITY=1 to compare with deep-sort-realtime"

if ENABLED:
    from deep_sort_realtime.deepsort_tracker import DeepSort

BOX2D = [20.0, 30.0, 40.0, 60.0]
FRAME_KEYS = "frame_index pipeline_inference_time_ms num_detections"
FRAME_KEYS = FRAME_KEYS + " num_tracks counts newly_counted_track_ids"
FRAME_KEYS = set((FRAME_KEYS + " detections tracks maxn").split())
SUMMARY_KEYS = "processed_frames final_counts counted_track_ids"
SUMMARY_KEYS = SUMMARY_KEYS + " counted_display_ids track_class_names"
SUMMARY_KEYS = set((SUMMARY_KEYS + " final_maxn pipeline_timing").split())
TIMING_KEYS = "scope total_seconds mean_ms_per_frame median_ms_per_frame"
TIMING_KEYS = set((TIMING_KEYS + " p95_ms_per_frame pipeline_fps").split())
TIMING_KEYS = TIMING_KEYS | {"first_frame_ms"}
TRACK_KEYS = "track_id display_id class_name box_xyxy length_frames"
TRACK_KEYS = set((TRACK_KEYS + " time_since_update counted").split())
DETECTION_KEYS = {"box_xyxy", "score", "class_name"}

NUM_FRAMES, H, W = 40, 320, 320
EMPTY_FRAME = 30
BACKGROUND_STEP = 6
NAMES = ["0", "Fish", "crab", "person"]
COUNT_CLASSES = ["fish", "crab"]
COLORS = {1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0)}
BAND_HEIGHT = 80
FEATURES = np.eye(4, dtype=np.float32)
Capture = namedtuple("Capture", "get")


def make_track(track_id, state, time_since_update, class_name, box2D=BOX2D):
    measurement = kalman.to_measurement(np.array(box2D, np.float32))
    args = (track_id, state, kalman.initiate(measurement), 1, 1)
    return Track(*args, time_since_update, [], class_name, 0.9)


def run_counting(tracks_per_frame, count_classes, min_track_frames=5):
    counting, newly_counted = build_counting(count_classes), []
    for tracks in tracks_per_frame:
        args = (counting, tracks, min_track_frames)
        counting, newly_counted_ids = update_counting(*args)
        newly_counted.append(newly_counted_ids)
    return counting, newly_counted


def test_track_matched_in_four_frames_is_not_counted():
    fish = make_track("1", CONFIRMED, 0, "fish")
    counting, newly_counted = run_counting([[fish]] * 4, ["fish"])
    assert counting.counts == {"fish": 0}
    assert newly_counted == [[]] * 4
    assert counting.lengths == {"1": 4}


def test_track_is_counted_once_at_five_matched_frames():
    fish = make_track("1", CONFIRMED, 0, "fish")
    counting, newly_counted = run_counting([[fish]] * 8, ["fish"])
    assert counting.counts == {"fish": 1}
    assert newly_counted == [[], [], [], [], ["1"], [], [], []]
    assert counting.display_ids == {"1": 1}


def test_missed_track_is_not_counted_until_matched_again():
    tentative = make_track("1", TENTATIVE, 0, "fish")
    missed = make_track("1", CONFIRMED, 1, "fish")
    matched = make_track("1", CONFIRMED, 0, "fish")
    frames = [[tentative]] * 5 + [[missed], [matched]]
    counting, newly_counted = run_counting(frames, ["fish"])
    assert newly_counted == [[]] * 6 + [["1"]]
    assert counting.lengths == {"1": 6}


def test_tentative_track_is_never_counted():
    tentative = make_track("1", TENTATIVE, 0, "fish")
    counting, newly_counted = run_counting([[tentative]] * 8, ["fish"])
    assert counting.counts == {"fish": 0}
    assert counting.display_ids == {}
    assert counting.lengths == {"1": 8}


def test_tentative_frames_count_towards_the_length():
    tentative = make_track("1", TENTATIVE, 0, "fish")
    confirmed = make_track("1", CONFIRMED, 0, "fish")
    frames = [[tentative]] * 3 + [[confirmed]] * 2
    _, newly_counted = run_counting(frames, ["fish"])
    assert newly_counted == [[], [], [], [], ["1"]]


def test_missed_frames_do_not_count_towards_the_length():
    matched = make_track("1", CONFIRMED, 0, "fish")
    missed_once = make_track("1", CONFIRMED, 1, "fish")
    missed_twice = make_track("1", CONFIRMED, 2, "fish")
    frames = [[matched]] * 3 + [[missed_once], [missed_twice]]
    frames = frames + [[matched]] * 2
    counting, newly_counted = run_counting(frames, ["fish"])
    assert newly_counted == [[]] * 6 + [["1"]]
    assert counting.lengths == {"1": 5}


def test_class_filter_is_case_insensitive_and_keeps_order():
    boxes2D = [[0, 0, 9, 9], [1, 1, 8, 8], [2, 2, 7, 7], [3, 3, 6, 6]]
    boxes2D = np.array(boxes2D, np.int32)
    class_names = ["Fish", "CRAB", "person", "fish"]
    scores = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    args = (boxes2D, class_names, scores, ["fish", "Crab"])
    kept_boxes2D, kept_names, kept_scores = select_classes(*args)
    np.testing.assert_array_equal(kept_boxes2D, boxes2D[[0, 1, 3]])
    assert list(kept_names) == ["Fish", "CRAB", "fish"]
    np.testing.assert_allclose(kept_scores, [0.9, 0.8, 0.6])


def test_class_filter_of_an_empty_frame_is_empty():
    args = (np.zeros((0, 4), np.int32), [], np.zeros(0), ["fish"])
    kept_boxes2D, kept_names, kept_scores = select_classes(*args)
    assert np.asarray(kept_boxes2D).shape == (0, 4)
    assert list(kept_names) == []
    assert np.asarray(kept_scores).shape == (0,)


def test_counts_use_the_configured_class_spelling():
    fish = make_track("1", CONFIRMED, 0, "FISH")
    counting, _ = run_counting([[fish]] * 5, ["Fish"])
    assert counting.counts == {"Fish": 1}


def test_display_ids_count_per_class_in_track_order():
    track_classes = [("4", "fish"), ("2", "crab"), ("7", "fish")]
    tracks = []
    for track_id, class_name in track_classes:
        tracks.append(make_track(track_id, CONFIRMED, 0, class_name))
    counting, newly_counted = run_counting([tracks] * 5, COUNT_CLASSES)
    assert newly_counted[4] == ["4", "2", "7"]
    assert counting.display_ids == {"4": 1, "2": 1, "7": 2}
    assert list(counting.counts.items()) == [("fish", 2), ("crab", 1)]


def test_counted_ids_are_sorted_as_strings():
    tracks = [make_track("2", CONFIRMED, 0, "fish")]
    tracks.append(make_track("10", CONFIRMED, 0, "fish"))
    counting, _ = run_counting([tracks] * 5, ["fish"])
    max_counts = count_frame_classes([], ["fish"])
    summary = build_summary(counting, max_counts, [1.0])
    assert summary["counted_track_ids"] == ["10", "2"]


def test_class_is_taken_at_counting_time():
    as_fish = make_track("1", CONFIRMED, 0, "fish")
    as_crab = make_track("1", CONFIRMED, 0, "crab")
    frames = [[as_fish]] * 4 + [[as_crab], [as_fish]]
    counting, newly_counted = run_counting(frames, COUNT_CLASSES)
    assert counting.counts == {"fish": 0, "crab": 1}
    assert newly_counted[4] == ["1"]
    assert counting.class_names == {"1": "fish"}


def test_class_names_hold_the_class_of_the_last_match():
    as_fish = make_track("1", CONFIRMED, 0, "fish")
    as_crab = make_track("1", CONFIRMED, 0, "crab")
    missed_as_fish = make_track("1", CONFIRMED, 1, "fish")
    frames = [[as_fish]] * 4 + [[as_crab], [missed_as_fish]]
    counting, _ = run_counting(frames, COUNT_CLASSES)
    assert counting.class_names == {"1": "crab"}


def test_track_of_an_uncounted_class_is_never_counted():
    person = make_track("1", CONFIRMED, 0, "person")
    counting, _ = run_counting([[person]] * 6, ["fish"])
    assert counting.counts == {"fish": 0}
    assert counting.lengths == {"1": 6}
    assert counting.class_names == {"1": "person"}


def test_update_counting_leaves_its_input_unchanged():
    counting = build_counting(["fish"])
    fish = make_track("1", CONFIRMED, 0, "fish")
    for _ in range(5):
        updated, _ = update_counting(counting, [fish], 1)
    assert counting == build_counting(["fish"])
    assert updated.counts == {"fish": 1}


def test_frame_classes_are_counted_case_insensitively():
    class_names = ["fish", "Fish", "crab", "person"]
    counts = count_frame_classes(class_names, COUNT_CLASSES)
    assert counts == {"fish": 2, "crab": 1}


def test_max_counts_keep_the_largest_frame_count_per_class():
    frames = [["fish", "fish", "crab"], ["fish"], ["crab", "CRAB", "crab"]]
    max_counts = count_frame_classes([], COUNT_CLASSES)
    for class_names in frames + [[]]:
        frame_counts = count_frame_classes(class_names, COUNT_CLASSES)
        max_counts = update_max_counts(max_counts, frame_counts)
    assert max_counts == {"fish": 2, "crab": 3}


def build_record_tracks():
    tracks = [make_track("1", CONFIRMED, 0, "fish")]
    tracks.append(make_track("2", TENTATIVE, 0, "fish"))
    missed_box2D = [50.0, 60.0, 30.0, 20.0]
    tracks.append(make_track("3", CONFIRMED, 3, "crab", missed_box2D))
    return tracks


def test_track_records_hold_confirmed_tracks_including_missed_ones():
    tracks = build_record_tracks()
    counting, _ = run_counting([tracks[:1]] * 5, COUNT_CLASSES)
    records = to_track_records(tracks, counting)
    assert [record["track_id"] for record in records] == ["1", "3"]
    for record in records:
        assert set(record) == TRACK_KEYS
    assert records[0]["display_id"] == 1 and records[0]["counted"] is True
    assert records[0]["length_frames"] == 5
    assert records[1]["display_id"] is None
    assert records[1]["counted"] is False
    assert records[1]["time_since_update"] == 3
    assert records[1]["length_frames"] == 0


def test_track_records_give_the_kalman_box_in_corners():
    tracks = build_record_tracks()
    records = to_track_records(tracks, build_counting(COUNT_CLASSES))
    x_min, y_min, width, height = compute_box2D(tracks[2])
    expected = [x_min, y_min, x_min + width, y_min + height]
    np.testing.assert_allclose(records[1]["box_xyxy"], expected)
    assert all(type(value) is float for value in records[1]["box_xyxy"])


def build_counted_frame():
    feature = FEATURES[0]
    box2D = np.array([10.0, 20.0, 30.0, 40.0], np.float32)
    detections = [Detection(box2D, 0.75, "fish", feature)]
    tracks = build_record_tracks()
    counting, newly_counted = run_counting([tracks[:1]] * 5, COUNT_CLASSES)
    max_counts = {"fish": 1, "crab": 0}
    args = (detections, tracks, counting, newly_counted[-1], max_counts)
    return CountedFrame(np.zeros((8, 8, 3), np.uint8), *args)


def test_frame_record_follows_the_reference_layout():
    record = build_frame_record(12, 3.5, build_counted_frame())
    assert set(record) == FRAME_KEYS
    assert record["frame_index"] == 12
    assert record["pipeline_inference_time_ms"] == 3.5
    assert record["num_detections"] == 1
    assert record["num_tracks"] == 2
    assert record["counts"] == {"fish": 1, "crab": 0}
    assert record["newly_counted_track_ids"] == ["1"]
    assert record["maxn"] == {"fish": 1, "crab": 0}
    detection = record["detections"][0]
    assert set(detection) == DETECTION_KEYS
    np.testing.assert_allclose(detection["box_xyxy"], [10, 20, 40, 60])
    assert detection["class_name"] == "fish"
    assert len(record["tracks"]) == 2


def test_frame_record_does_not_share_the_counts():
    counted_frame = build_counted_frame()
    record = build_frame_record(0, 1.0, counted_frame)
    record["counts"]["fish"] = 99
    assert counted_frame.counting.counts["fish"] == 1


def test_frame_record_and_summary_are_json_serializable():
    counted_frame = build_counted_frame()
    record = build_frame_record(0, np.float64(1.0), counted_frame)
    args = (counted_frame.counting, counted_frame.max_counts, [1.0])
    payload = dict(frames=[record], **build_summary(*args))
    assert json.loads(json.dumps(payload))["frames"][0]["frame_index"] == 0


def test_summary_follows_the_reference_layout():
    counted_frame = build_counted_frame()
    args = (counted_frame.counting, counted_frame.max_counts)
    summary = build_summary(*args, [100.0, 10.0, 20.0, 30.0])
    assert set(summary) == SUMMARY_KEYS
    assert set(summary["pipeline_timing"]) == TIMING_KEYS
    assert summary["processed_frames"] == 4
    assert summary["final_counts"] == {"fish": 1, "crab": 0}
    assert summary["counted_display_ids"] == {"1": 1}
    assert summary["track_class_names"] == {"1": "fish"}
    assert summary["final_maxn"] == {"fish": 1, "crab": 0}


def test_timing_summarizes_every_frame():
    times_ms = [100.0, 10.0, 20.0, 30.0]
    timing = summarize_timing(times_ms)
    assert timing["total_seconds"] == pytest.approx(0.16)
    assert timing["mean_ms_per_frame"] == pytest.approx(40.0)
    assert timing["median_ms_per_frame"] == pytest.approx(25.0)
    p95 = float(np.percentile(times_ms, 95))
    assert timing["p95_ms_per_frame"] == pytest.approx(p95)
    assert timing["pipeline_fps"] == pytest.approx(4 / 0.16)
    assert timing["first_frame_ms"] == pytest.approx(100.0)


def test_timing_of_no_frames_has_no_rates():
    timing = summarize_timing([])
    assert timing["total_seconds"] == 0.0
    names = ["mean_ms_per_frame", "median_ms_per_frame", "p95_ms_per_frame"]
    for name in names + ["pipeline_fps", "first_frame_ms"]:
        assert timing[name] is None


FPS_CASES = [(0.0, 25.0), (-1.0, 25.0), (float("nan"), 25.0), (29.97, 29.97)]


@pytest.mark.parametrize("fps, expected", FPS_CASES)
def test_read_fps_falls_back_to_25(fps, expected):
    assert read_fps(Capture(lambda name: fps)) == pytest.approx(expected)


def build_scene(frame_index):
    x = 10 + 3 * frame_index
    boxes2D = [[x, 260, x + 40, 300], [x, 20, x + 40, 60]]
    boxes2D.append([x + 100, 100, x + 140, 140])
    labels = [3, 1, 2]
    if frame_index >= 20:
        boxes2D.append([x + 50, 180, x + 90, 220])
        labels.append(1)
    if frame_index == EMPTY_FRAME:
        boxes2D, labels = [], []
    return boxes2D, labels


def build_synthetic_frames():
    frames = []
    for frame_index in range(NUM_FRAMES):
        level = BACKGROUND_STEP * frame_index
        frame = np.full((H, W, 3), level, np.uint8)
        for box2D, label in zip(*build_scene(frame_index)):
            x_min, y_min, x_max, y_max = box2D
            frame[y_min:y_max, x_min:x_max] = COLORS[label]
        frames.append(frame)
    return frames


def write_synthetic_video(directory):
    path = str(directory / "synthetic.mp4")
    paz.video.from_frames(build_synthetic_frames(), path, 10)
    return path


def build_scripted_detect():
    calls = []

    def detect(image):
        boxes2D, labels = build_scene(len(calls))
        calls.append(image.shape)
        boxes2D = np.array(boxes2D, np.int32).reshape(-1, 4)
        scores = np.full(len(labels), 0.9, np.float32)
        return boxes2D, np.array(labels, np.int32), scores

    return detect


def embed_by_band(image, boxes2D):
    bands = (np.asarray(boxes2D)[:, 1] // BAND_HEIGHT).astype(int)
    return FEATURES[bands]


def measure_background(frame):
    return float(np.mean(np.asarray(frame)[:12]))


def read_background_levels(path):
    # the codec shifts flat colours a few levels, so seeked frames are
    # compared with the same frames decoded sequentially, not with the drawn
    capture, levels = cv2.VideoCapture(path), []
    received, frame = capture.read()
    while received:
        levels.append(measure_background(frame))
        received, frame = capture.read()
    capture.release()
    return levels


def test_read_frames_seeks_and_stops_at_max_frames(tmp_path):
    path = write_synthetic_video(tmp_path)
    expected = read_background_levels(path)[10:15]
    frames = list(read_frames(open_video(path, 10), 5))
    assert len(frames) == 5
    levels = [measure_background(frame) for frame in frames]
    np.testing.assert_allclose(levels, expected, atol=1.0)


def test_read_frames_reads_to_the_end_in_RGB(tmp_path):
    capture = open_video(write_synthetic_video(tmp_path), 0)
    frames = list(read_frames(capture, None))
    assert len(frames) == NUM_FRAMES
    assert np.asarray(frames[0]).shape == (H, W, 3)
    red = np.asarray(frames[0])[30:50, 15:45].mean(axis=(0, 1))
    assert red[0] > 200 and red[2] < 60


def test_pair_frames_pairs_each_frame_with_its_result():
    frames = [np.full((4, 4, 3), index, np.uint8) for index in range(3)]

    def track(images):
        for image in images:
            yield int(image[0, 0, 0])

    pairs = list(pair_frames(track, (frame for frame in frames)))
    assert [result for _, result in pairs] == [0, 1, 2]
    for (image, _), frame in zip(pairs, frames):
        np.testing.assert_array_equal(image, frame)


def test_time_steps_yields_each_step_with_its_duration():
    timed = list(time_steps(iter(["a", "b"])))
    assert [step for step, _ in timed] == ["a", "b"]
    assert all(time_ms >= 0.0 for _, time_ms in timed)


def test_build_detect_maps_labels_to_names_and_filters_classes():
    def detect(image):
        boxes2D = np.array([[0, 0, 9, 9], [1, 1, 8, 8], [2, 2, 7, 7]])
        return boxes2D, np.array([1, 3, 2]), np.array([0.9, 0.8, 0.7])

    image = np.zeros((10, 10, 3), np.uint8)
    names = ["0", "fish", "person", "Crab"]
    detect_names = build_detect(detect, names, ["FISH", "crab"])
    boxes2D, class_names, scores = detect_names(image)
    np.testing.assert_array_equal(boxes2D, [[0, 0, 9, 9], [1, 1, 8, 8]])
    assert list(class_names) == ["fish", "Crab"]
    np.testing.assert_allclose(scores, [0.9, 0.8])


def build_draw_tracks():
    corners = [(20, 60), (180, 60), (20, 180), (180, 180)]
    states = [(CONFIRMED, 0), (TENTATIVE, 0), (CONFIRMED, 6), (CONFIRMED, 5)]
    tracks = []
    for index, (corner, state) in enumerate(zip(corners, states)):
        box2D = [*corner, 40, 40]
        args = (str(index + 1), *state, "fish", box2D)
        tracks.append(make_track(*args))
    return tracks


def is_border_drawn(drawn, image, box2D):
    x_min, y_min, _, height = box2D
    point = (y_min + height // 2, x_min)
    return not np.array_equal(drawn[point], image[point])


def test_draw_tracks_draws_confirmed_tracks_missed_at_most_max_missed():
    image = np.zeros((240, 320, 3), np.uint8)
    tracks = build_draw_tracks()
    drawn = np.asarray(draw_tracks(image, tracks, {"1": 1}, 5))
    assert drawn.shape == image.shape
    assert not image.any()
    flags = []
    for track in tracks:
        box2D = [int(round(value)) for value in compute_box2D(track)]
        flags.append(is_border_drawn(drawn, image, box2D))
    assert flags == [True, False, False, True]


def test_draw_counts_keeps_the_image_shape():
    image = np.zeros((240, 320, 3), np.uint8)
    drawn = np.asarray(draw_counts(image, {"fish": 2, "crab": 1}))
    assert drawn.shape == image.shape
    assert drawn.any() and not image.any()


def test_write_json_writes_the_payload(tmp_path):
    path = tmp_path / "tracks.json"
    payload = {"final_counts": {"fish": 2}, "frames": []}
    write_json(str(path), payload)
    with open(path) as json_file:
        assert json.load(json_file) == payload


def run_synthetic_video(directory):
    detect = build_detect(build_scripted_detect(), NAMES, COUNT_CLASSES)
    args = (detect, embed_by_band, Config(), False)
    track = DeepSORT(*args)
    capture = open_video(write_synthetic_video(directory), 0)
    pairs = pair_frames(track, read_frames(capture, None))
    return list(count_tracked_frames(pairs, COUNT_CLASSES, 5))


def collect_newly_counted(newly_counted_per_frame):
    newly_counted = {}
    for frame_index, track_ids in enumerate(newly_counted_per_frame):
        if track_ids:
            newly_counted[frame_index] = track_ids
    return newly_counted


def test_synthetic_video_is_counted_end_to_end(tmp_path):
    counted_frames = run_synthetic_video(tmp_path)
    assert len(counted_frames) == NUM_FRAMES
    ids = [frame.newly_counted_ids for frame in counted_frames]
    assert collect_newly_counted(ids) == {4: ["1", "2"], 24: ["3"]}
    last = counted_frames[-1]
    summary = build_summary(last.counting, last.max_counts, [1.0] * 40)
    assert summary["final_counts"] == {"fish": 2, "crab": 1}
    assert summary["counted_track_ids"] == ["1", "2", "3"]
    assert summary["counted_display_ids"] == {"1": 1, "2": 1, "3": 2}
    assert summary["final_maxn"] == {"fish": 2, "crab": 1}


def test_synthetic_video_frame_records(tmp_path):
    counted_frames = run_synthetic_video(tmp_path)
    records = []
    for frame_index, counted_frame in enumerate(counted_frames):
        records.append(build_frame_record(frame_index, 1.0, counted_frame))
    assert records[3]["counts"] == {"fish": 0, "crab": 0}
    assert records[4]["counts"] == {"fish": 1, "crab": 1}
    assert records[-1]["counts"] == {"fish": 2, "crab": 1}
    assert records[EMPTY_FRAME]["num_detections"] == 0
    assert records[EMPTY_FRAME]["num_tracks"] == 3
    assert records[EMPTY_FRAME + 1]["num_detections"] == 3
    class_names = [record["class_name"] for record in records[0]["detections"]]
    assert class_names == ["Fish", "crab"]
    json.dumps(records)


def to_oracle_detection(detection):
    box2D = [float(value) for value in detection.box2D]
    return (box2D, float(detection.score), detection.class_name)


def to_track(oracle_track):
    args = (str(oracle_track.track_id), oracle_track.state, None)
    args = args + (oracle_track.hits, oracle_track.age)
    class_name = oracle_track.get_det_class()
    args = args + (oracle_track.time_since_update, [], class_name)
    return Track(*args, oracle_track.det_conf)


def run_oracle_counting(counted_frames):
    kwargs = dict(max_age=30, n_init=3, max_iou_distance=0.7)
    kwargs.update(max_cosine_distance=0.2, nn_budget=None, embedder=None)
    oracle, counting = DeepSort(**kwargs), build_counting(COUNT_CLASSES)
    newly_counted, counts = [], []
    for counted_frame in counted_frames:
        detections = counted_frame.detections
        inputs = [to_oracle_detection(detection) for detection in detections]
        embeds = [np.asarray(detection.feature) for detection in detections]
        oracle_tracks = oracle.update_tracks(inputs, embeds=embeds)
        tracks = [to_track(oracle_track) for oracle_track in oracle_tracks]
        counting, track_ids = update_counting(counting, tracks, 5)
        newly_counted.append(track_ids)
        counts.append(dict(counting.counts))
    return newly_counted, counts


@pytest.mark.skipif(not ENABLED, reason=REASON)
def test_synthetic_counts_equal_the_oracle_tracker_counts(tmp_path):
    counted_frames = run_synthetic_video(tmp_path)
    oracle_newly_counted, oracle_counts = run_oracle_counting(counted_frames)
    ids = [frame.newly_counted_ids for frame in counted_frames]
    counts = [dict(frame.counting.counts) for frame in counted_frames]
    assert ids == oracle_newly_counted
    assert counts == oracle_counts
