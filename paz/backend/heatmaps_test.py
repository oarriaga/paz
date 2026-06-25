import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.backend import heatmaps


def test_filter_peaks_keeps_single_maximum():
    heatmap = np.zeros((1, 1, 5, 5))
    heatmap[0, 0, 2, 2] = 1.0
    peaks = heatmaps.filter_peaks(heatmap)
    assert peaks[0, 0, 2, 2] == 1.0
    assert peaks.sum() == 1.0


def test_top_k_detections_shape():
    rng = np.random.default_rng(0)
    maps = rng.random((1, 17, 16, 16))
    tags = rng.random((1, 17, 16, 16, 1))
    detections = heatmaps.top_k_detections(maps, tags, k=5)
    assert detections.shape == (17, 5, 4)


def build_two_person_detections():
    keypoint0 = [[10, 10, 0.9, 0.0], [50, 50, 0.9, 10.0]]
    keypoint1 = [[12, 12, 0.9, 0.1], [52, 52, 0.9, 10.1]]
    return np.array([keypoint0, keypoint1], dtype=float)


def test_group_by_tag_separates_two_people():
    detections = build_two_person_detections()
    people = heatmaps.group_by_tag(detections, [0, 1], 1.0, 0.2)
    assert len(people) == 2


def test_group_by_tag_assigns_both_keypoints():
    detections = build_two_person_detections()
    people = heatmaps.group_by_tag(detections, [0, 1], 1.0, 0.2)
    for person in people:
        assert (person[:, 2] > 0).sum() == 2


def test_group_by_tag_groups_nearby_tags_together():
    detections = build_two_person_detections()
    people = heatmaps.group_by_tag(detections, [0, 1], 1.0, 0.2)
    first_person = people[0]
    assert np.allclose(first_person[0, :2], [10, 10])
    assert np.allclose(first_person[1, :2], [12, 12])
