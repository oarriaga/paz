import os
import json

import numpy as np
import pytest

import paz
import paz.datasets.ozfish as ozfish


S3 = "s3://ozfish-bucket/frames/"


def build_box(left, top, width, height):
    return dict(class_id=3, left=left, top=top, width=width, height=height)


def build_payload(W, H, boxes):
    image_size = [dict(width=W, height=H, depth=3)]
    return dict(annotations=boxes, image_size=image_size)


def build_record(source_key, payload_key, payload):
    return {"source-ref": S3 + source_key, payload_key: payload}


def to_line(record):
    return json.dumps(record) + "\n"


def write_frame(root, folder, name, W, H):
    folder = paz.directory.make(os.path.join(root, "frames_labelled", folder))
    image_path = os.path.join(folder, name)
    paz.image.write(image_path, np.zeros((H, W, 3), "uint8"))
    return os.path.normpath(image_path)


def write_manifest(root, name, lines):
    folder = paz.directory.make(os.path.join(root, "manifests"))
    with open(os.path.join(folder, name), "w") as manifest_file:
        manifest_file.write("".join(lines))


def write_frames(root):
    names = ["E01.MP4.31568.png-176-1.png", "E01.MP4.31600.png-9-1.png"]
    names = names + ["E02.MP4.100.png", "clip.JPG-12-1.jpg"]
    names = names + ["E03.AVI.7.png-2-1.png", "E04.MP4.9.png-3-1.png"]
    folders = 2 * ["batch_1"] + 2 * ["batch_2/deep"] + 2 * ["batch_2"]
    sizes = [(8, 4), (16, 8), (8, 4), (16, 8), (8, 8), (8, 4)]
    paths = []
    for folder, name, (W, H) in zip(folders, names, sizes):
        paths.append(write_frame(root, folder, name, W, H))
    return paths


def build_first_manifest():
    boxes = [build_box(8, 4, 12, 4), build_box(28, 0, 8, 8)]
    boxes = boxes + [build_box(6, 2, 4, 2)]
    decoy = dict(annotations=[], class_map={"0": "fish"})
    record = {"source-ref": S3 + "E01.MP4.31568.png", "decoy": decoy}
    record["fish-labels"] = build_payload(32, 16, boxes)
    record["late"] = build_payload(32, 16, [])
    no_payload = {"source-ref": S3 + "ghost.png", "fish-labels": decoy}
    empty = build_record("E01.MP4.31600.png", "job", build_payload(32, 32, []))
    lines = [to_line(record), "\n", to_line(no_payload), "   \n"]
    return lines + [to_line(empty)]


def build_second_manifest():
    boxes = [build_box(1, 1, 0, 2), build_box(10, 1, 2, 2)]
    boxes = boxes + [build_box(-1, -1, 2, 2), build_box(1, 1, 2, 2)]
    no_size = dict(annotations=boxes, image_size=[dict(width=0)])
    below = build_payload(32, 16, [build_box(0, 12, 4, 8)])
    uneven = build_payload(16, 32, [build_box(2, 4, 4, 8)])
    full = build_payload(32, 16, [build_box(0, 0, 32, 16)])
    records = [build_record("E02.MP4.100.png", "fish-labels", no_size)]
    records.append(build_record("clip.JPG", "labels-2", below))
    records.append(build_record("E03.AVI.7.png", "fish-labels", uneven))
    records.append(build_record("E04.MP4.9.png", "fish-labels", full))
    return [to_line(record) for record in records]


def write_dataset(root):
    paths = write_frames(root)
    write_manifest(root, "b.manifest", build_second_manifest())
    write_manifest(root, "a.manifest", build_first_manifest())
    return paths


def build_expected():
    first = [[2, 1, 5, 2, 0], [7, 0, 8, 2, 0], [1.5, 0.5, 2.5, 1, 0]]
    third = [[0, 0, 1, 1, 0], [1, 1, 3, 3, 0]]
    rest = [[[0, 6, 2, 8, 0]], [[1, 1, 3, 3, 0]], [[0, 0, 8, 4, 0]]]
    return [first, [], third] + rest


def write_rows(root, frames, source_keys):
    for folder, name in frames:
        write_frame(root, folder, name, 8, 4)
    records = []
    for source_key in source_keys:
        payload = build_payload(8, 4, [build_box(1, 1, 2, 2)])
        records.append(build_record(source_key, "fish-labels", payload))
    lines = [to_line(record) for record in records]
    write_manifest(root, "a.manifest", lines)


def normalize(paths):
    return [os.path.normpath(path) for path in paths]


def load_by_path(root):
    by_path = {}
    for split in ["train", "validation"]:
        image_paths, detections = ozfish.load(root, split)
        by_path.update(zip(normalize(image_paths), detections))
    return by_path


def build_frame_path(root, folder, name):
    frame_path = os.path.join(root, "frames_labelled", folder, name)
    return os.path.normpath(frame_path)


def assert_detections(detections, expected):
    expected = np.array(expected, dtype="float32").reshape(-1, 5)
    assert isinstance(detections, np.ndarray)
    assert detections.dtype == np.float32
    assert detections.shape == expected.shape
    np.testing.assert_array_equal(detections, expected)


def assert_all_detections(detections, expected):
    assert len(detections) == len(expected)
    for image_detections, image_expected in zip(detections, expected):
        assert_detections(image_detections, image_expected)


def test_load_train_takes_first_shuffled_rows_in_order(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    image_paths, detections = ozfish.load(root, "train")
    order = [3, 1, 2, 4, 0]
    assert normalize(image_paths) == [paths[index] for index in order]
    expected = build_expected()
    assert_all_detections(detections, [expected[index] for index in order])


def test_load_validation_takes_remaining_rows(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    image_paths, detections = ozfish.load(root, "validation")
    assert normalize(image_paths) == [paths[5]]
    assert_all_detections(detections, [build_expected()[5]])


def test_load_scales_clamps_and_zeroes_classes(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    detections = load_by_path(root)[paths[0]]
    assert_detections(detections, build_expected()[0])


def test_load_keeps_empty_annotations_as_negative(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    assert_detections(load_by_path(root)[paths[1]], [])


def test_load_uses_real_size_when_manifest_size_is_zero(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    detections = load_by_path(root)[paths[2]]
    assert_detections(detections, [[0, 0, 1, 1, 0], [1, 1, 3, 3, 0]])


def test_load_clamps_box_below_image(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    assert_detections(load_by_path(root)[paths[3]], [[0, 6, 2, 8, 0]])


def test_load_scales_width_and_height_separately(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    assert_detections(load_by_path(root)[paths[4]], [[1, 1, 3, 3, 0]])


def test_load_skips_rows_without_payload(tmp_path):
    root = str(tmp_path)
    paths = write_dataset(root)
    assert sorted(load_by_path(root)) == sorted(paths)


def test_load_cuts_key_after_first_image_extension(tmp_path):
    root = str(tmp_path)
    frames = [("b", "E01.MP4.5.png-176-1.jpg"), ("b", "clip.JPG-12-1.png")]
    write_rows(root, frames, ["E01.MP4.5.png", "clip.JPG"])
    expected_0 = build_frame_path(root, "b", "E01.MP4.5.png-176-1.jpg")
    expected_1 = build_frame_path(root, "b", "clip.JPG-12-1.png")
    assert sorted(load_by_path(root)) == sorted([expected_0, expected_1])


def test_load_takes_first_frame_of_a_repeated_key(tmp_path):
    root = str(tmp_path)
    frames = [("a", "X.png-1-1.png"), ("b", "X.png-2-1.png"), ("b", "Y.png")]
    write_rows(root, frames, ["X.png", "Y.png"])
    labels_path = os.path.join(root, "frames_labelled", "a", "X.png-0.txt")
    with open(labels_path, "w") as labels_file:
        labels_file.write("not an image")
    outside = paz.directory.make(os.path.join(root, "a"))
    outside_path = os.path.join(outside, "X.png-0-0.png")
    paz.image.write(outside_path, np.zeros((4, 8, 3), "uint8"))
    expected_x = build_frame_path(root, "a", "X.png-1-1.png")
    expected_y = build_frame_path(root, "b", "Y.png")
    assert sorted(load_by_path(root)) == sorted([expected_x, expected_y])


def test_load_sorts_frames_by_folder_parts(tmp_path):
    root = str(tmp_path)
    frames = [("a-b", "X.png-1-1.png"), ("a", "X.png-2-1.png"), ("a", "Y.png")]
    write_rows(root, frames, ["X.png", "Y.png"])
    expected_x = build_frame_path(root, "a", "X.png-2-1.png")
    expected_y = build_frame_path(root, "a", "Y.png")
    assert sorted(load_by_path(root)) == sorted([expected_x, expected_y])


def test_load_sorts_frames_case_sensitively(tmp_path):
    root = str(tmp_path)
    frames = [("B", "X.png-1-1.png"), ("a", "X.png-2-1.png"), ("a", "Y.png")]
    write_rows(root, frames, ["X.png", "Y.png"])
    expected_x = build_frame_path(root, "B", "X.png-1-1.png")
    expected_y = build_frame_path(root, "a", "Y.png")
    assert sorted(load_by_path(root)) == sorted([expected_x, expected_y])


def test_load_sorts_manifests_case_sensitively(tmp_path):
    root = str(tmp_path)
    write_rows(root, [("a", "X.png"), ("a", "Y.png")], ["X.png"])
    payload = build_payload(8, 4, [build_box(1, 1, 2, 2)])
    record = build_record("Y.png", "fish-labels", payload)
    write_manifest(root, "B.manifest", [to_line(record)])
    train_paths = ozfish.load(root, "train")[0]
    assert normalize(train_paths) == [build_frame_path(root, "a", "X.png")]


def test_load_indexes_every_image_extension(tmp_path):
    root = str(tmp_path)
    frames = [("a", "Z.PNG"), ("a", "W.jpeg"), ("a", "V.bmp")]
    write_rows(root, frames, ["Z.PNG", "W.jpeg", "V.bmp"])
    expected = [build_frame_path(root, "a", name) for _, name in frames]
    assert sorted(load_by_path(root)) == sorted(expected)


def test_load_matches_keys_case_sensitively(tmp_path):
    root = str(tmp_path)
    write_rows(root, [("a", "X.png"), ("a", "Y.png")], ["x.png", "Y.png"])
    with pytest.raises(FileNotFoundError):
        ozfish.load(root, "train")


def test_load_returns_string_paths(tmp_path):
    root = str(tmp_path)
    write_dataset(root)
    image_paths = ozfish.load(root, "train")[0]
    assert all(isinstance(path, str) for path in image_paths)


def test_load_keeps_one_validation_row(tmp_path):
    root = str(tmp_path)
    frames = [("a", "X.png"), ("a", "Y.png")]
    write_rows(root, frames, ["X.png", "Y.png"])
    train_paths = ozfish.load(root, "train")[0]
    validation_paths = ozfish.load(root, "validation")[0]
    assert normalize(train_paths) == [build_frame_path(root, "a", "Y.png")]
    expected_x = build_frame_path(root, "a", "X.png")
    assert normalize(validation_paths) == [expected_x]


def test_load_raises_listing_every_missing_image(tmp_path):
    root = str(tmp_path)
    frames = [("a", "X.png")]
    write_rows(root, frames, ["X.png", "lost_1.png", "lost_2.png"])
    with pytest.raises(FileNotFoundError) as error:
        ozfish.load(root, "train")
    assert "lost_1.png" in str(error.value)
    assert "lost_2.png" in str(error.value)


def test_load_raises_on_missing_folders(tmp_path):
    with pytest.raises(ValueError):
        ozfish.load(str(tmp_path / "missing"), "train")


def test_load_raises_with_one_usable_row(tmp_path):
    root = str(tmp_path)
    write_rows(root, [("a", "X.png")], ["X.png"])
    with pytest.raises(ValueError):
        ozfish.load(root, "train")


@pytest.mark.parametrize("split", ["val", "valid"])
def test_load_rejects_unknown_split(tmp_path, split):
    root = str(tmp_path)
    write_dataset(root)
    with pytest.raises(ValueError):
        ozfish.load(root, split)


def test_get_class_names():
    assert ozfish.get_class_names() == ["fish"]
