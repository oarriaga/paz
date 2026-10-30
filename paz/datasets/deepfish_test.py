import os

import numpy as np
import pytest

import paz
import paz.datasets.deepfish as deepfish


WRONG_FIELD_COUNTS = ["0 0.5 0.5 0.25\n", "0 0.5 0.5 0.25 0.5 1\n"]


def build_fake_download(root):
    def fake_download(*args, **kwargs):
        return root
    return fake_download


def load_split(monkeypatch, root, split):
    monkeypatch.setattr(deepfish, "download", build_fake_download(root))
    return deepfish.load(split)


def load_random_split(monkeypatch, root, split):
    monkeypatch.setattr(deepfish, "download", build_fake_download(root))
    return deepfish.load_random_split(split)


LOAD_FUNCTIONS = [load_split, load_random_split]
LOAD_NAMES = ["load", "load_random_split"]


def write_image(image_path, W, H):
    paz.image.write(image_path, np.zeros((H, W, 3), "uint8"))


def write_labels(label_path, lines):
    with open(label_path, "w") as label_file:
        label_file.write("".join(lines))


def write_unlabelled(folder, stem, W, H):
    paz.directory.make(folder)
    write_image(os.path.join(folder, stem + ".jpg"), W, H)


def write_sample(folder, stem, W, H, lines):
    write_unlabelled(folder, stem, W, H)
    write_labels(os.path.join(folder, stem + ".txt"), lines)


def build_line(index):
    return f"0 {(index + 0.5) / 16} 0.5 0.0625 0.5\n"


def build_row(index):
    return [index, 1, index + 1, 3, 0]


def write_indexed_images(root, folders, stems):
    for index, (folder, stem) in enumerate(zip(folders, stems)):
        folder = os.path.join(root, folder)
        write_sample(folder, stem, 16, 4, [build_line(index)])


def build_thirteen_folders():
    video_0 = 4 * ["video_0/train"] + 2 * ["video_0/valid"]
    video_1 = 5 * ["video_1/train"] + 2 * ["video_1/valid"]
    return video_0 + video_1


def write_thirteen_images(root):
    folders = build_thirteen_folders()
    stems = [f"{index:02d}" for index in range(13)]
    write_indexed_images(root, folders, stems)


def write_duplicate_stems(root):
    folders = ["video_0/train", "video_0/valid"]
    folders = folders + ["video_1/train", "video_1/valid"]
    write_indexed_images(root, folders, ["a", "b", "a", "b"])


def build_path(root, folder, stem):
    return os.path.normpath(os.path.join(root, folder, stem + ".jpg"))


def build_thirteen_paths(root, indices):
    folders = build_thirteen_folders()
    paths = []
    for index in indices:
        paths.append(build_path(root, folders[index], f"{index:02d}"))
    return paths


def normalize(paths):
    return [os.path.normpath(path) for path in paths]


def load_one_image(monkeypatch, root, W, H, lines):
    write_sample(os.path.join(root, "video_0", "train"), "a", W, H, lines)
    detections = load_split(monkeypatch, root, "train")[1]
    return detections[0]


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


def test_load_pairs_labels_by_stem(tmp_path, monkeypatch):
    root = str(tmp_path)
    folder = os.path.join(root, "video_0", "train")
    write_sample(folder, "a", 8, 4, ["0 0.25 0.5 0.25 0.5\n"])
    write_labels(os.path.join(folder, "b.txt"), ["0 0.5 0.5 0.25 0.5\n"])
    write_sample(folder, "c", 8, 4, ["0 0.75 0.5 0.25 0.5\n"])
    image_paths, detections = load_split(monkeypatch, root, "train")
    expected_a = build_path(root, "video_0/train", "a")
    expected_c = build_path(root, "video_0/train", "c")
    expected_paths = [expected_a, expected_c]
    assert normalize(image_paths) == expected_paths
    assert_all_detections(detections, [[[1, 1, 3, 3, 0]], [[5, 1, 7, 3, 0]]])


def test_load_treats_missing_label_file_as_negative(tmp_path, monkeypatch):
    root = str(tmp_path)
    folder = os.path.join(root, "video_0", "train")
    write_unlabelled(folder, "a", 8, 4)
    write_sample(folder, "b", 8, 4, ["0 0.5 0.5 0.25 0.5\n"])
    image_paths, detections = load_split(monkeypatch, root, "train")
    expected_a = build_path(root, "video_0/train", "a")
    expected_b = build_path(root, "video_0/train", "b")
    expected_paths = [expected_a, expected_b]
    assert normalize(image_paths) == expected_paths
    assert_all_detections(detections, [[], [[3, 1, 5, 3, 0]]])


def test_load_keeps_empty_label_file_as_negative(tmp_path, monkeypatch):
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, [])
    assert_detections(detections, [])


def test_load_uses_decoded_image_size(tmp_path, monkeypatch):
    root = str(tmp_path)
    folder = os.path.join(root, "video_0", "train")
    write_sample(folder, "a", 8, 4, ["0 0.5 0.5 0.25 0.5\n"])
    write_sample(folder, "b", 16, 8, ["0 0.5 0.5 0.25 0.5\n"])
    detections = load_split(monkeypatch, root, "train")[1]
    assert_all_detections(detections, [[[3, 1, 5, 3, 0]], [[6, 2, 10, 6, 0]]])


def test_load_keeps_fractional_pixels(tmp_path, monkeypatch):
    lines = ["0 0.5 0.5 0.375 0.25\n"]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    assert_detections(detections, [[2.5, 1.5, 5.5, 2.5, 0]])


def test_load_sets_every_class_to_zero(tmp_path, monkeypatch):
    lines = ["3 0.5 0.5 0.25 0.5\n", "1 0.25 0.5 0.25 0.5\n"]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    assert_detections(detections, [[3, 1, 5, 3, 0], [1, 1, 3, 3, 0]])


def test_load_clamps_to_image_size(tmp_path, monkeypatch):
    left = "0 0.0625 0.5 0.25 0.5\n"
    right = "0 0.9375 0.5 0.25 0.5\n"
    top = "0 0.5 0.125 0.25 0.5\n"
    bottom = "0 0.5 0.875 0.25 0.5\n"
    lines = [left, right, top, bottom]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    expected_x = [[0, 1, 1.5, 3, 0], [6.5, 1, 8, 3, 0]]
    expected_y = [[3, 0, 5, 1.5, 0], [3, 2.5, 5, 4, 0]]
    assert_detections(detections, expected_x + expected_y)


def test_load_drops_boxes_outside_image(tmp_path, monkeypatch):
    outside_x = ["0 1.5 0.5 0.25 0.5\n", "0 -0.5 0.5 0.25 0.5\n"]
    outside_y = ["0 0.5 1.5 0.25 0.5\n", "0 0.5 -0.5 0.25 0.5\n"]
    on_edge = ["0 1.125 0.5 0.25 0.5\n"]
    lines = outside_x + outside_y + on_edge + ["0 0.5 0.5 0.25 0.5\n"]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    assert_detections(detections, [[3, 1, 5, 3, 0]])


def test_load_drops_zero_and_negative_size_boxes(tmp_path, monkeypatch):
    zero = ["0 0.5 0.5 0.0 0.5\n", "0 0.5 0.5 0.25 0.0\n"]
    negative = ["0 0.5 0.5 -0.25 0.5\n", "0 0.5 0.5 0.25 -0.5\n"]
    lines = zero + negative + ["0 0.5 0.5 0.25 0.5\n"]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    assert_detections(detections, [[3, 1, 5, 3, 0]])


def test_load_drops_boxes_empty_in_float32(tmp_path, monkeypatch):
    lines = ["0 0.5 0.5 1e-9 0.5\n"]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    assert_detections(detections, [])


def test_load_skips_blank_lines(tmp_path, monkeypatch):
    first = "0 0.5 0.5 0.25 0.5\n"
    last = "0 0.25 0.5 0.25 0.5"
    lines = ["\n", first, "   \n", "\n", last]
    detections = load_one_image(monkeypatch, str(tmp_path), 8, 4, lines)
    assert_detections(detections, [[3, 1, 5, 3, 0], [1, 1, 3, 3, 0]])


@pytest.mark.parametrize("line", WRONG_FIELD_COUNTS)
def test_load_raises_on_wrong_field_count(tmp_path, monkeypatch, line):
    with pytest.raises(ValueError):
        load_one_image(monkeypatch, str(tmp_path), 8, 4, [line])


@pytest.mark.parametrize("load_function", LOAD_FUNCTIONS, ids=LOAD_NAMES)
def test_load_rejects_unknown_split(tmp_path, monkeypatch, load_function):
    root = str(tmp_path)
    write_thirteen_images(root)
    with pytest.raises(ValueError):
        load_function(monkeypatch, root, "val")


def test_load_orders_train_images_by_path(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_indexed_images(root, ["video_1/train"], ["a"])
    write_sample(os.path.join(root, "video_0", "train"), "b", 16, 4, [])
    write_sample(os.path.join(root, "video_0", "valid"), "c", 16, 4, [])
    image_paths, detections = load_split(monkeypatch, root, "train")
    expected_b = build_path(root, "video_0/train", "b")
    expected_a = build_path(root, "video_1/train", "a")
    assert normalize(image_paths) == [expected_b, expected_a]
    assert_all_detections(detections, [[], [build_row(0)]])


def test_load_validation_reads_valid_folders(tmp_path, monkeypatch):
    root = str(tmp_path)
    folders = ["video_0/train", "video_0/valid", "video_1/valid"]
    write_indexed_images(root, folders, ["a", "b", "c"])
    image_paths, detections = load_split(monkeypatch, root, "validation")
    expected_b = build_path(root, "video_0/valid", "b")
    expected_c = build_path(root, "video_1/valid", "c")
    assert normalize(image_paths) == [expected_b, expected_c]
    assert_all_detections(detections, [[build_row(1)], [build_row(2)]])


def test_load_keeps_duplicate_stems_across_videos(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_indexed_images(root, ["video_0/train", "video_1/train"], ["a", "a"])
    image_paths, detections = load_split(monkeypatch, root, "train")
    expected_0 = build_path(root, "video_0/train", "a")
    expected_1 = build_path(root, "video_1/train", "a")
    assert normalize(image_paths) == [expected_0, expected_1]
    assert_all_detections(detections, [[build_row(0)], [build_row(1)]])


def test_load_random_split_train_members_in_path_order(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_thirteen_images(root)
    image_paths, detections = load_random_split(monkeypatch, root, "train")
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    assert normalize(image_paths) == build_thirteen_paths(root, indices)
    expected = [[build_row(index)] for index in indices]
    assert_all_detections(detections, expected)


def test_load_random_split_validation_members(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_thirteen_images(root)
    split = "validation"
    image_paths, detections = load_random_split(monkeypatch, root, split)
    assert normalize(image_paths) == build_thirteen_paths(root, [9, 11])
    assert_all_detections(detections, [[build_row(9)], [build_row(11)]])


def test_load_random_split_keeps_one_validation_image(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_duplicate_stems(root)
    split = "validation"
    image_paths, detections = load_random_split(monkeypatch, root, split)
    assert normalize(image_paths) == [build_path(root, "video_0/valid", "b")]
    assert_all_detections(detections, [[build_row(1)]])


def test_load_random_split_sorts_paths_with_slashes(tmp_path, monkeypatch):
    root = str(tmp_path)
    folders = ["v1/train", "v1/train", "v10/train", "v10/train"]
    write_indexed_images(root, folders, ["a", "b", "c", "d"])
    image_paths, detections = load_random_split(monkeypatch, root, "train")
    expected_a = build_path(root, "v1/train", "a")
    expected_c = build_path(root, "v10/train", "c")
    expected_d = build_path(root, "v10/train", "d")
    assert normalize(image_paths) == [expected_a, expected_c, expected_d]
    expected = [[build_row(0)], [build_row(2)], [build_row(3)]]
    assert_all_detections(detections, expected)


def test_load_random_split_keeps_duplicate_stems(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_duplicate_stems(root)
    image_paths, detections = load_random_split(monkeypatch, root, "train")
    expected_0 = build_path(root, "video_0/train", "a")
    expected_2 = build_path(root, "video_1/train", "a")
    expected_3 = build_path(root, "video_1/valid", "b")
    expected_paths = [expected_0, expected_2, expected_3]
    assert normalize(image_paths) == expected_paths
    expected = [[build_row(0)], [build_row(2)], [build_row(3)]]
    assert_all_detections(detections, expected)
