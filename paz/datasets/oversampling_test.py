import numpy as np
import pytest

import paz.datasets.oversampling as oversampling


def build_detections(rows):
    return np.array(rows, dtype="float32").reshape(-1, 5)


def build_dataset():
    image_paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    common = build_detections([[0, 0, 1, 1, 0]])
    mixed = build_detections([[0, 0, 2, 2, 1], [1, 1, 3, 3, 0]])
    negative = build_detections([])
    rare = build_detections([[2, 2, 4, 4, 2]])
    return image_paths, [common, mixed, negative, rare]


def repeat_dataset(rare_class_args, num_copies):
    image_paths, detections = build_dataset()
    args = (image_paths, detections, rare_class_args, num_copies)
    return oversampling.repeat_rare_images(*args)


def assert_all_detections(detections, expected):
    assert len(detections) == len(expected)
    for image_detections, image_expected in zip(detections, expected):
        np.testing.assert_array_equal(image_detections, image_expected)


def test_repeat_rare_images_appends_copies_after_originals():
    image_paths, detections, copy_indices = repeat_dataset([1, 2], 2)
    originals = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert image_paths == originals + ["b.jpg", "b.jpg", "d.jpg", "d.jpg"]
    assert copy_indices == [0, 0, 0, 0, 1, 2, 1, 2]


def test_repeat_rare_images_copies_every_box_of_the_image():
    detections = repeat_dataset([1, 2], 2)[1]
    common, mixed, negative, rare = build_dataset()[1]
    expected = [common, mixed, negative, rare, mixed, mixed, rare, rare]
    assert_all_detections(detections, expected)


def test_repeat_rare_images_matches_one_rare_class():
    image_paths, detections, copy_indices = repeat_dataset([2], 3)
    originals = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert image_paths == originals + ["d.jpg", "d.jpg", "d.jpg"]
    assert copy_indices == [0, 0, 0, 0, 1, 2, 3]


def test_repeat_rare_images_leaves_inputs_unchanged():
    image_paths, detections = build_dataset()
    args = (image_paths, detections, [1, 2], 2)
    oversampling.repeat_rare_images(*args)
    assert image_paths == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert_all_detections(detections, build_dataset()[1])


def test_repeat_rare_images_raises_without_rare_images():
    with pytest.raises(ValueError):
        repeat_dataset([3], 2)
