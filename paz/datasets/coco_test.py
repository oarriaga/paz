import os

import numpy as np
import pytest

import paz
import paz.datasets.coco as coco


SPLITS = [("train", "train", 0), ("val", "valid", 1), ("test", "test", 2)]


def build_image(image_id, file_name, W, H):
    return dict(id=image_id, file_name=file_name, width=W, height=H)


def build_annotation(image_id, category_id, box2D):
    return dict(image_id=image_id, category_id=category_id, bbox=box2D)


def build_category(category_id, name, supercategory):
    return dict(id=category_id, name=name, supercategory=supercategory)


def build_fish_categories():
    fishes = build_category(0, "fishes", "none")
    fish = build_category(1, "fish", "fishes")
    return [fishes, fish]


def build_categories(category_ids, names):
    categories = []
    for category_id, name in zip(category_ids, names):
        categories.append(build_category(category_id, name, "fishes"))
    return categories


def build_images(image_ids, file_names):
    images = []
    for image_id, file_name in zip(image_ids, file_names):
        images.append(build_image(image_id, file_name, 20, 10))
    return images


def build_annotations(category_ids):
    annotations = []
    for index, category_id in enumerate(category_ids):
        box2D = [index, index, 1, 1]
        annotations.append(build_annotation(1, category_id, box2D))
    return annotations


def build_row(index, label):
    return [index, index, index + 1, index + 1, label]


def build_fish_annotations(boxes):
    return [build_annotation(1, 1, box2D) for box2D in boxes]


def write_image(image_path):
    paz.image.write(image_path, np.zeros((2, 3, 3), "uint8"))


def write_split(root, folder, images, annotations, categories):
    split_path = os.path.join(root, folder)
    os.makedirs(split_path)
    for image in images:
        write_image(os.path.join(split_path, image["file_name"]))
    data = dict(images=images, annotations=annotations, categories=categories)
    json_path = os.path.join(split_path, "_annotations.coco.json")
    paz.file.write_json(data, json_path)


def write_one_image(root, folder, annotations, categories):
    image = build_image(1, "a.jpg", 20, 10)
    write_split(root, folder, [image], annotations, categories)


def write_split_folders(root):
    categories = build_fish_categories()
    for _, folder, offset in SPLITS:
        image = build_image(1, f"{folder}.jpg", 20, 10)
        annotation = build_annotation(1, 1, [offset, 0, 1, 1])
        write_split(root, folder, [image], [annotation], categories)


def load_one_image(root, split, class_names):
    detections = coco.load(root, split, class_names)[1]
    return detections[0]


def load_fish_boxes(root, boxes):
    annotations = build_fish_annotations(boxes)
    write_one_image(root, "train", annotations, build_fish_categories())
    return load_one_image(root, "train", ["fish"])


def assert_detections(detections, expected):
    expected = np.array(expected, dtype="float32").reshape(-1, 5)
    assert isinstance(detections, np.ndarray)
    assert detections.dtype == np.float32
    assert detections.shape == expected.shape
    np.testing.assert_array_equal(detections, expected)


def test_load_class_names_skips_none_supercategory(tmp_path):
    root = str(tmp_path)
    crab = build_category(2, "crab", "fishes")
    categories = build_fish_categories() + [crab]
    write_one_image(root, "train", [], categories)
    assert coco.load_class_names(root) == ["fish", "crab"]


def test_load_class_names_keeps_real_class_at_id_zero(tmp_path):
    root = str(tmp_path)
    fish = build_category(0, "fish", "animals")
    crab = build_category(1, "crab", "animals")
    write_one_image(root, "train", [], [crab, fish])
    assert coco.load_class_names(root) == ["fish", "crab"]


def test_load_class_names_keeps_missing_supercategory(tmp_path):
    root = str(tmp_path)
    cod = dict(id=2, name="cod")
    categories = build_fish_categories() + [cod]
    write_one_image(root, "train", [], categories)
    assert coco.load_class_names(root) == ["fish", "cod"]


def test_load_class_names_sorts_by_id(tmp_path):
    root = str(tmp_path)
    categories = build_categories([10, 1, 5], ["a", "b", "c"])
    write_one_image(root, "train", [], categories)
    assert coco.load_class_names(root) == ["b", "c", "a"]


def test_load_class_names_keeps_categories_without_boxes(tmp_path):
    root = str(tmp_path)
    categories = build_categories([1, 2, 3], ["cod", "hake", "eel"])
    write_one_image(root, "train", build_annotations([2]), categories)
    assert coco.load_class_names(root) == ["cod", "hake", "eel"]


def test_load_class_names_reads_train_split(tmp_path):
    root = str(tmp_path)
    train_categories = build_categories([1, 2], ["cod", "hake"])
    write_one_image(root, "train", [], train_categories)
    write_one_image(root, "valid", [], build_categories([1], ["crab"]))
    write_one_image(root, "test", [], build_categories([1], ["eel"]))
    assert coco.load_class_names(root) == ["cod", "hake"]


def test_load_returns_path_lists_and_float_detections(tmp_path):
    root = str(tmp_path)
    annotations = build_fish_annotations([[1.5, 2.25, 3.5, 4.0]])
    write_one_image(root, "train", annotations, build_fish_categories())
    image_paths, detections = coco.load(root, "train", ["fish"])
    assert image_paths == [os.path.join(root, "train", "a.jpg")]
    assert isinstance(detections, list) and len(detections) == 1
    assert_detections(detections[0], [[1.5, 2.25, 5.0, 6.25, 0]])


@pytest.mark.parametrize("split, folder, offset", SPLITS)
def test_load_reads_split_folder(tmp_path, split, folder, offset):
    root = str(tmp_path)
    write_split_folders(root)
    image_paths, detections = coco.load(root, split, ["fish"])
    assert image_paths == [os.path.join(root, folder, f"{folder}.jpg")]
    assert_detections(detections[0], [[offset, 0, offset + 1, 1, 0]])


def test_load_orders_images_by_id(tmp_path):
    root = str(tmp_path)
    images = build_images([10, 2, 1], ["a.jpg", "b.jpg", "c.jpg"])
    annotation_10 = build_annotation(10, 1, [2, 2, 1, 1])
    annotation_1 = build_annotation(1, 1, [1, 1, 1, 1])
    annotations = [annotation_10, annotation_1]
    write_split(root, "train", images, annotations, build_fish_categories())
    image_paths, detections = coco.load(root, "train", ["fish"])
    file_names = [os.path.basename(path) for path in image_paths]
    assert file_names == ["c.jpg", "b.jpg", "a.jpg"]
    assert_detections(detections[0], [[1, 1, 2, 2, 0]])
    assert_detections(detections[1], [])
    assert_detections(detections[2], [[2, 2, 3, 3, 0]])


def test_load_keeps_image_boxes_in_annotation_order(tmp_path):
    boxes = [[4, 4, 2, 2], [1, 1, 2, 2]]
    detections = load_fish_boxes(str(tmp_path), boxes)
    assert_detections(detections, [[4, 4, 6, 6, 0], [1, 1, 3, 3, 0]])


def test_load_keeps_negative_image_as_empty(tmp_path):
    detections = load_fish_boxes(str(tmp_path), [])
    assert_detections(detections, [])


def test_load_drops_crowd_boxes(tmp_path):
    root = str(tmp_path)
    unflagged = build_annotation(1, 1, [1, 1, 2, 2])
    crowd = dict(unflagged, bbox=[7, 7, 2, 2], iscrowd=1)
    not_crowd = dict(unflagged, bbox=[4, 4, 2, 2], iscrowd=0)
    annotations = [unflagged, crowd, not_crowd]
    write_one_image(root, "train", annotations, build_fish_categories())
    detections = load_one_image(root, "train", ["fish"])
    assert_detections(detections, [[1, 1, 3, 3, 0], [4, 4, 6, 6, 0]])


def test_load_keeps_all_crowd_image_as_empty(tmp_path):
    root = str(tmp_path)
    crowd = dict(build_annotation(1, 1, [1, 1, 2, 2]), iscrowd=1)
    write_one_image(root, "train", [crowd], build_fish_categories())
    assert_detections(load_one_image(root, "train", ["fish"]), [])


def test_load_clamps_to_json_size(tmp_path):
    boxes = [[15, 5, 10, 10], [-5, -2, 10, 4], [18, 2, 5, 5]]
    detections = load_fish_boxes(str(tmp_path), boxes)
    expected = [[15, 5, 20, 10, 0], [0, 0, 5, 2, 0], [18, 2, 20, 7, 0]]
    assert_detections(detections, expected)


def test_load_drops_boxes_outside_image(tmp_path):
    outside = [[25, 2, 5, 5], [-10, 2, 5, 5], [2, 12, 3, 3], [2, -8, 3, 3]]
    edges = [[20, 2, 5, 5], [-1, 2, 1, 5], [2, 10, 3, 3], [2, -1, 3, 1]]
    boxes = outside + edges + [[1, 1, 2, 2]]
    detections = load_fish_boxes(str(tmp_path), boxes)
    assert_detections(detections, [[1, 1, 3, 3, 0]])


def test_load_drops_zero_and_negative_area_boxes(tmp_path):
    degenerate = [[3, 3, 0, 4], [3, 3, 4, 0], [3, 3, -2, 4], [3, 3, 4, -2]]
    boxes = degenerate + [[1, 1, 2, 2]]
    detections = load_fish_boxes(str(tmp_path), boxes)
    assert_detections(detections, [[1, 1, 3, 3, 0]])


def test_load_keeps_all_degenerate_image_as_empty(tmp_path):
    boxes = [[3, 3, 0, 4], [25, 2, 5, 5]]
    detections = load_fish_boxes(str(tmp_path), boxes)
    assert_detections(detections, [])


def test_load_closes_category_id_gaps(tmp_path):
    root = str(tmp_path)
    categories = build_categories([10, 1, 5], ["a", "b", "c"])
    annotations = build_annotations([10, 1, 5])
    write_one_image(root, "train", annotations, categories)
    class_names = coco.load_class_names(root)
    detections = load_one_image(root, "train", class_names)
    expected = [build_row(0, 2), build_row(1, 0), build_row(2, 1)]
    assert_detections(detections, expected)


def test_load_keeps_real_class_at_id_zero(tmp_path):
    root = str(tmp_path)
    fish = build_category(0, "fish", "animals")
    crab = build_category(1, "crab", "animals")
    annotations = build_annotations([0, 1, 0])
    write_one_image(root, "train", annotations, [fish, crab])
    class_names = coco.load_class_names(root)
    detections = load_one_image(root, "train", class_names)
    expected = [build_row(0, 0), build_row(1, 1), build_row(2, 0)]
    assert_detections(detections, expected)


def test_load_drops_none_supercategory_boxes(tmp_path):
    root = str(tmp_path)
    group_0 = build_category(0, "fish", "none")
    fish = build_category(1, "fish", "animals")
    group_2 = build_category(2, "fish", "none")
    annotations = build_annotations([0, 1, 2])
    write_one_image(root, "train", annotations, [group_0, fish, group_2])
    detections = load_one_image(root, "train", ["fish"])
    assert_detections(detections, [build_row(1, 0)])


def test_load_keeps_missing_supercategory_boxes(tmp_path):
    root = str(tmp_path)
    cod = dict(id=1, name="cod")
    annotations = build_annotations([1])
    write_one_image(root, "train", annotations, [cod])
    detections = load_one_image(root, "train", ["cod"])
    assert_detections(detections, [build_row(0, 0)])


def test_load_keeps_class_index_of_boxless_category(tmp_path):
    root = str(tmp_path)
    categories = build_categories([1, 2], ["cod", "hake"])
    write_one_image(root, "train", build_annotations([2]), categories)
    class_names = coco.load_class_names(root)
    detections = load_one_image(root, "train", class_names)
    assert_detections(detections, [build_row(0, 1)])


def test_load_maps_split_to_train_classes_by_name(tmp_path):
    root = str(tmp_path)
    train_categories = build_categories([1, 2], ["cod", "hake"])
    valid_categories = build_categories([7, 8], ["hake", "cod"])
    write_one_image(root, "train", [], train_categories)
    write_one_image(root, "valid", build_annotations([7, 8]), valid_categories)
    class_names = coco.load_class_names(root)
    detections = load_one_image(root, "val", class_names)
    assert_detections(detections, [build_row(0, 1), build_row(1, 0)])


def test_load_drops_boxes_of_unknown_category(tmp_path):
    root = str(tmp_path)
    train_categories = build_categories([1, 2], ["cod", "hake"])
    valid_categories = build_categories([1, 2, 3], ["cod", "crab", "hake"])
    write_one_image(root, "train", [], train_categories)
    annotations = build_annotations([2, 42, 3])
    write_one_image(root, "valid", annotations, valid_categories)
    class_names = coco.load_class_names(root)
    detections = load_one_image(root, "val", class_names)
    assert_detections(detections, [build_row(2, 1)])


def test_load_keeps_image_whose_boxes_are_all_unknown(tmp_path):
    root = str(tmp_path)
    categories = build_categories([1], ["crab"])
    write_one_image(root, "train", build_annotations([1]), categories)
    image_paths, detections = coco.load(root, "train", ["cod"])
    assert image_paths == [os.path.join(root, "train", "a.jpg")]
    assert_detections(detections[0], [])


def test_load_follows_given_class_order(tmp_path):
    root = str(tmp_path)
    categories = build_categories([1, 2], ["cod", "hake"])
    write_one_image(root, "train", build_annotations([1, 2]), categories)
    detections = load_one_image(root, "train", ["hake", "cod"])
    assert_detections(detections, [build_row(0, 1), build_row(1, 0)])
