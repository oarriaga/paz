"""DeepSORT application tests.

They run on CPU with a scripted detector and a one-hot embedder, so they
need no download.
"""
import numpy as np

from paz.models.tracking.deep_sort.application import DeepSORT
from paz.models.tracking.deep_sort.application import TrackedFrame
from paz.models.tracking.deep_sort.application import draw_track_boxes
from paz.models.tracking.deep_sort import CONFIRMED
from paz.models.tracking.deep_sort import TENTATIVE
from paz.models.tracking.deep_sort import Config
from paz.models.tracking.deep_sort import Detection
from paz.models.tracking.deep_sort import build_tracker
from paz.models.tracking.deep_sort import update
from paz.models.tracking.deep_sort import compute_box2D
from paz.models.tracking.deep_sort import Track
from paz.models.tracking.deep_sort import kalman

H, W = 120, 160
BAND_HEIGHT = 40
FEATURES = np.eye(8, dtype=np.float32)


def build_frames(num_frames):
    frames = []
    for frame_index in range(num_frames):
        frame = np.zeros((H, W, 3), np.uint8)
        # the fake detector reads its script entry from this pixel
        frame[0, 0, 0] = frame_index
        frames.append(frame)
    return frames


def build_detect(script):
    def detect(image):
        boxes2D, class_names, scores = script[int(image[0, 0, 0])]
        boxes2D = np.array(boxes2D, np.int32).reshape(-1, 4)
        return boxes2D, list(class_names), np.array(scores, np.float32)

    return detect


def build_moving_script(num_frames):
    script = {}
    for frame_index in range(num_frames):
        x = 10 + 2 * frame_index
        boxes2D = [[x, 5, x + 30, 35], [x + 60, 85, x + 90, 115]]
        script[frame_index] = (boxes2D, ["fish", "Crab"], [0.9, 0.8])
    return script


def embed_by_band(image, boxes2D):
    # one feature per horizontal band, so each object keeps its feature
    bands = (np.asarray(boxes2D)[:, 1] // BAND_HEIGHT).astype(int)
    return FEATURES[bands]


def build_recording_embed(calls):
    def embed(image, boxes2D):
        calls.append(boxes2D)
        return embed_by_band(image, boxes2D)

    return embed


def refuse_embed(image, boxes2D):
    raise AssertionError("embed was called on a frame without detections")


def track_script(script, embed, draw):
    args = (build_detect(script), embed, Config(), draw)
    track = DeepSORT(*args)
    return list(track(build_frames(len(script))))


def to_expected_detections(step):
    detections = []
    for box, class_name, score in zip(*step):
        x_min, y_min, x_max, y_max = box
        box2D = [x_min, y_min, x_max - x_min, y_max - y_min]
        box2D = np.array(box2D, np.float32)
        feature = FEATURES[int(y_min // BAND_HEIGHT)]
        args = (box2D, float(np.float32(score)), class_name, feature)
        detections.append(Detection(*args))
    return detections


def run_direct_update(script):
    tracker, tracks_per_frame = build_tracker(), []
    for frame_index in range(len(script)):
        detections = to_expected_detections(script[frame_index])
        tracker = update(Config(), tracker, detections)
        tracks_per_frame.append(tracker.tracks)
    return tracks_per_frame


def describe(track):
    args = (track.state, track.hits, track.age, track.time_since_update)
    return (track.id, *args, track.class_name)


def test_deep_sort_yields_one_tracked_frame_per_frame():
    results = track_script(build_moving_script(4), embed_by_band, False)
    assert len(results) == 4
    for result in results:
        assert isinstance(result, TrackedFrame)
        assert len(result.detections) == 2


def test_deep_sort_reads_frames_from_a_generator():
    script = build_moving_script(3)
    detect = build_detect(script)
    track = DeepSORT(detect, embed_by_band, Config(), False)
    frames = (frame for frame in build_frames(3))
    assert len(list(track(frames))) == 3


def test_deep_sort_never_embeds_a_frame_without_detections():
    empty = ([], [], [])
    script = {0: empty, 1: empty, 2: empty}
    results = track_script(script, refuse_embed, False)
    assert len(results) == 3
    for result in results:
        assert result.detections == []
        assert result.tracks == []


def test_deep_sort_never_embeds_a_frame_of_only_invalid_boxes():
    flat = ([[10, 10, 40, 10]], ["fish"], [0.9])
    thin = ([[10, 10, 10, 40]], ["fish"], [0.9])
    results = track_script({0: flat, 1: thin}, refuse_embed, False)
    assert [result.detections for result in results] == [[], []]


def test_deep_sort_converts_corner_boxes_to_float32_xywh():
    calls = []
    script = {0: ([[10, 20, 50, 80]], ["fish"], [0.9])}
    result = track_script(script, build_recording_embed(calls), False)[0]
    boxes2D = np.asarray(calls[0])
    assert boxes2D.dtype == np.float32
    np.testing.assert_array_equal(boxes2D, [[10.0, 20.0, 40.0, 60.0]])
    box2D = np.asarray(result.detections[0].box2D)
    assert box2D.dtype == np.float32
    np.testing.assert_array_equal(box2D, [10.0, 20.0, 40.0, 60.0])


def test_deep_sort_embeds_only_valid_boxes_in_detector_order():
    boxes2D = [[0, 0, 10, 10], [5, 5, 5, 20], [30, 50, 60, 90], [9, 9, 30, 9]]
    class_names = ["fish", "crab", "lobster", "fish"]
    script = {0: (boxes2D, class_names, [0.9, 0.8, 0.7, 0.6])}
    calls = []
    result = track_script(script, build_recording_embed(calls), False)[0]
    assert len(calls) == 1
    expected = [[0.0, 0.0, 10.0, 10.0], [30.0, 50.0, 30.0, 40.0]]
    np.testing.assert_array_equal(np.asarray(calls[0]), expected)
    names = [detection.class_name for detection in result.detections]
    assert names == ["fish", "lobster"]


def test_deep_sort_attaches_one_feature_row_per_valid_box():
    script = {0: ([[0, 0, 10, 10], [30, 50, 60, 90]], ["a", "b"], [1, 1])}
    result = track_script(script, embed_by_band, False)[0]
    features = [detection.feature for detection in result.detections]
    np.testing.assert_array_equal(features[0], FEATURES[0])
    np.testing.assert_array_equal(features[1], FEATURES[1])


def test_deep_sort_passes_class_names_and_scores_through():
    script = build_moving_script(1)
    result = track_script(script, embed_by_band, False)[0]
    names = [detection.class_name for detection in result.detections]
    scores = [detection.score for detection in result.detections]
    assert names == ["fish", "Crab"]
    np.testing.assert_allclose(scores, [0.9, 0.8], atol=1e-6)
    assert [track.class_name for track in result.tracks] == ["fish", "Crab"]


def test_deep_sort_tracks_equal_a_direct_tracker_update():
    script = build_moving_script(6)
    results = track_script(script, embed_by_band, False)
    expected = run_direct_update(script)
    for result, tracks in zip(results, expected):
        described = [describe(track) for track in result.tracks]
        assert described == [describe(track) for track in tracks]
        for track, expected_track in zip(result.tracks, tracks):
            expected_box2D = compute_box2D(expected_track)
            box2D = compute_box2D(track)
            np.testing.assert_allclose(box2D, expected_box2D, atol=1e-4)


def test_deep_sort_confirms_tracks_on_the_third_hit():
    results = track_script(build_moving_script(3), embed_by_band, False)
    states = [[track.state for track in result.tracks] for result in results]
    assert states == [[TENTATIVE] * 2, [TENTATIVE] * 2, [CONFIRMED] * 2]
    assert [track.id for track in results[2].tracks] == ["1", "2"]


def test_deep_sort_calls_draw_with_the_frame_and_its_tracks():
    calls = []

    def draw(image, tracks):
        calls.append((image, tracks))
        return np.full_like(image, 7)

    script = build_moving_script(2)
    results = track_script(script, embed_by_band, draw)
    frames = build_frames(2)
    assert len(calls) == 2
    for result, call, frame in zip(results, calls, frames):
        (tracked, drawn), (image, tracks) = result, call
        assert isinstance(tracked, TrackedFrame)
        assert tracks == tracked.tracks
        np.testing.assert_array_equal(image, frame)
        np.testing.assert_array_equal(drawn, np.full_like(frame, 7))


def test_deep_sort_default_draw_returns_an_image_of_the_frame_shape():
    results = track_script(build_moving_script(3), embed_by_band, None)
    for _, drawn in results:
        assert np.asarray(drawn).shape == (H, W, 3)
        assert np.asarray(drawn).dtype == np.uint8


def test_deep_sort_default_draw_marks_only_confirmed_tracks():
    results = track_script(build_moving_script(3), embed_by_band, None)
    frames = build_frames(3)
    np.testing.assert_array_equal(np.asarray(results[0][1]), frames[0])
    assert not np.array_equal(np.asarray(results[2][1]), frames[2])


def make_track(track_id, state, time_since_update, box2D):
    measurement = kalman.to_measurement(np.array(box2D, np.float32))
    args = (track_id, state, kalman.initiate(measurement), 3, 3)
    return Track(*args, time_since_update, [], "fish", 0.9)


def is_left_border_drawn(drawn, image, box2D):
    x_min, y_min, _, height = box2D
    point = (y_min + height // 2, x_min)
    return not np.array_equal(drawn[point], image[point])


def test_default_draw_skips_missed_and_tentative_tracks():
    corners = [(10, 40), (90, 40), (10, 100), (90, 100)]
    states = [(CONFIRMED, 0), (CONFIRMED, 1), (CONFIRMED, 6), (TENTATIVE, 0)]
    tracks, boxes2D = [], []
    for index, (corner, state) in enumerate(zip(corners, states)):
        boxes2D.append([*corner, 30, 30])
        tracks.append(make_track(str(index + 1), *state, boxes2D[-1]))
    image = np.zeros((H, W, 3), np.uint8)
    drawn = np.asarray(draw_track_boxes(image, tracks))
    assert drawn.shape == image.shape and not image.any()
    flags = []
    for box2D in boxes2D:
        flags.append(is_left_border_drawn(drawn, image, box2D))
    assert flags == [True, False, False, False]
