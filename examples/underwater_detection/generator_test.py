import os
import functools
import importlib.util

import numpy as np
import jax
import pytest

import paz
from paz.datasets import oversampling

MEAN = np.array([0.485, 0.456, 0.406], "float32")
STDV = np.array([0.229, 0.224, 0.225], "float32")
SIZES = [(40, 56), (48, 32)]
SIZE = 32


@functools.cache
def load_generator():
    path = os.path.join(os.path.dirname(__file__), "generator.py")
    name = "underwater_detection_generator"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_train_generator(seed, paths, detections, copy_indices, batch_size):
    key = jax.random.PRNGKey(seed)
    args = (key, paths, detections, copy_indices, batch_size, SIZE, 1)
    return load_generator().TrainGenerator(*args)


def build_validation_generator(paths, detections, batch_size):
    args = (paths, detections, batch_size, SIZE, 1)
    return load_generator().ValidationGenerator(*args)


def write_image(path, image):
    paz.image.write(path, image)
    return path


def write_constant(path, H, W, color):
    image = np.empty((H, W, 3), "uint8")
    image[:] = color
    return write_image(path, image)


def build_full_box(H, W, class_arg):
    return np.array([[0.0, 0.0, W, H, class_arg]], "float32")


def write_indexed_images(root, num_images):
    paths, detections = [], []
    for arg in range(num_images):
        H, W = SIZES[arg % 2]
        path = os.path.join(root, f"{arg:02d}.png")
        paths.append(write_constant(path, H, W, (10 * arg + 5, 30, 200)))
        detections.append(build_full_box(H, W, 0))
    return paths, detections


def write_red_rectangle(root):
    image = np.zeros((48, 64, 3), "uint8")
    image[12:36, 8:28, 0] = 255
    path = write_image(os.path.join(root, "red.png"), image)
    return path, np.array([[8.0, 12.0, 28.0, 36.0, 1.0]], "float32")


def write_texture(root):
    texture = np.random.RandomState(0).randint(0, 256, (40, 56, 3))
    path = os.path.join(root, "texture.png")
    return write_image(path, texture.astype("uint8"))


def read_epoch(generator, epoch):
    generator.epoch = epoch
    batches = [generator[arg] for arg in range(len(generator))]
    images = np.concatenate([np.asarray(batch[0]) for batch in batches])
    targets = np.concatenate([np.asarray(batch[1]) for batch in batches])
    return images, targets


def destandardize(images):
    return np.rint((np.asarray(images) * STDV + MEAN) * 255.0)


def read_colors(images):
    return destandardize(images)[:, SIZE // 2, SIZE // 2]


def read_indices(images):
    return ((read_colors(images)[:, 0] - 5) / 10).astype(int)


def compute_epoch_order(seed, epoch, num_samples):
    order_key = jax.random.split(jax.random.PRNGKey(seed))[0]
    epoch_key = jax.random.fold_in(order_key, epoch)
    return np.asarray(jax.random.permutation(epoch_key, num_samples))


def augment_copy(seed, path, detections, list_position):
    copy_key = jax.random.split(jax.random.PRNGKey(seed))[1]
    key = jax.random.fold_in(copy_key, list_position)
    args = (key, paz.image.load(path), detections, 0.35)
    return paz.detection.random_flip_enhance_blur(*args)


def compute_copy_color(seed, path, list_position):
    detections = np.full((1, 5), -1.0, "float32")
    image = augment_copy(seed, path, detections, list_position)[0]
    return tuple(np.asarray(image)[0, 0].astype(float))


def write_offline_copy(seed, path, detection, list_position):
    H, W = paz.image.get_size(paz.image.load(path))
    detections = paz.detection.normalize(detection, H, W)
    image, boxes = augment_copy(seed, path, detections, list_position)
    copy_path = write_image(path.replace(".png", "_copy.png"), image)
    boxes = paz.to_numpy(paz.detection.denormalize(boxes, H, W))
    return copy_path, boxes


def compute_red_corners(image):
    red = destandardize(image)
    is_red = (red[..., 0] - red[..., 1]) > 30
    rows, columns = np.nonzero(is_red)
    if len(rows) == 0:
        corners = None
    else:
        lower = [columns.min(), rows.min()]
        corners = np.array(lower + [columns.max() + 1, rows.max() + 1])
    return corners


def to_pixel_corners(box):
    center, size = np.asarray(box[:2]), np.asarray(box[2:4])
    return np.concatenate([center - size / 2, center + size / 2]) * SIZE


def test_train_length_drops_the_last_partial_batch(tmp_path):
    paths, detections = write_indexed_images(str(tmp_path), 10)
    generator = build_train_generator(0, paths, detections, [0] * 10, 4)
    assert len(generator) == 2
    for images, targets in [generator[0], generator[1]]:
        assert np.asarray(images).shape == (4, SIZE, SIZE, 3)
        assert np.asarray(targets).shape == (4, 1, 5)


def test_validation_length_keeps_the_last_partial_batch(tmp_path):
    paths, detections = write_indexed_images(str(tmp_path), 10)
    generator = build_validation_generator(paths, detections, 4)
    assert len(generator) == 3
    images, targets = generator[2]
    assert np.asarray(images).shape == (2, SIZE, SIZE, 3)
    assert np.asarray(targets).shape == (2, 1, 5)


def test_order_is_the_permutation_keyed_by_seed_and_epoch(tmp_path):
    paths, detections = write_indexed_images(str(tmp_path), 10)
    generator = build_train_generator(3, paths, detections, [0] * 10, 4)
    assert generator.epoch == 0
    for epoch in [0, 1, 5]:
        images, _ = read_epoch(generator, epoch)
        expected = compute_epoch_order(3, epoch, 10)[:8]
        assert np.array_equal(read_indices(images), expected)


def test_fresh_generators_with_one_seed_repeat_every_epoch(tmp_path):
    paths, detections = write_indexed_images(str(tmp_path), 9)
    args = (paths, detections, [0] * 9, 4)
    generator = build_train_generator(7, *args)
    epochs = [read_epoch(generator, epoch) for epoch in range(4)]
    resumed = build_train_generator(7, *args)
    for epoch in [3, 1, 2, 0]:
        images, targets = read_epoch(resumed, epoch)
        assert np.array_equal(images, epochs[epoch][0])
        assert np.array_equal(targets, epochs[epoch][1])


def test_epochs_and_seeds_change_the_order(tmp_path):
    paths, detections = write_indexed_images(str(tmp_path), 12)
    args = (paths, detections, [0] * 12, 4)
    generator = build_train_generator(0, *args)
    other_seed = build_train_generator(1, *args)
    epoch_0 = read_indices(read_epoch(generator, 0)[0])
    epoch_1 = read_indices(read_epoch(generator, 1)[0])
    seed_1 = read_indices(read_epoch(other_seed, 0)[0])
    assert sorted(epoch_0) == list(range(12))
    assert not np.array_equal(epoch_0, epoch_1)
    assert not np.array_equal(epoch_0, seed_1)


def test_batches_repeat_within_an_epoch(tmp_path):
    paths = [write_texture(str(tmp_path))] * 8
    detections = [build_full_box(40, 56, 0)] * 8
    generator = build_train_generator(2, paths, detections, [0] * 8, 4)
    generator.epoch = 1
    first = np.asarray(generator[1][0])
    assert not np.array_equal(np.asarray(generator[0][0]), first)
    generator.on_epoch_end()
    assert np.array_equal(np.asarray(generator[1][0]), first)


def test_batches_and_samples_get_distinct_keys(tmp_path):
    paths = [write_texture(str(tmp_path))] * 8
    detections = [build_full_box(40, 56, 0)] * 8
    generator = build_train_generator(2, paths, detections, [0] * 8, 4)
    images_0 = read_epoch(generator, 0)[0]
    images_1 = read_epoch(generator, 1)[0]
    assert not np.array_equal(images_0[:4], images_0[4:])
    assert not np.array_equal(images_0, images_1)
    for images in [images_0[:4], images_0[4:], images_1[:4]]:
        assert not np.all(images == images[:1])


def test_train_standardizes_with_imagenet_statistics(tmp_path):
    path = write_constant(str(tmp_path / "image.png"), 40, 56, (105, 30, 200))
    detections = [build_full_box(40, 56, 0)] * 4
    generator = build_train_generator(0, [path] * 4, detections, [0] * 4, 4)
    images = np.asarray(generator[0][0])
    expected = (np.array([105, 30, 200], "float32") / 255.0 - MEAN) / STDV
    assert images.dtype == np.float32
    assert images.shape == (4, SIZE, SIZE, 3)
    assert np.allclose(images, expected, atol=1e-5)


def test_train_targets_are_center_form_padded_with_minus_one(tmp_path):
    paths, detections = write_indexed_images(str(tmp_path), 6)
    detections[2] = np.repeat(detections[2], 3, axis=0)
    detections[4] = detections[4][:0]
    detections[5][:, 4] = 1.0
    generator = build_train_generator(4, paths, detections, [0] * 6, 3)
    for epoch in range(3):
        images, targets = read_epoch(generator, epoch)
        assert targets.shape == (6, 3, 5)
        for index, target in zip(read_indices(images), targets):
            num_boxes = len(detections[index])
            valid = target[:num_boxes]
            class_arg = 1.0 if index == 5 else 0.0
            expected = [0.5, 0.5, 1.0, 1.0, class_arg]
            assert np.allclose(valid, expected, atol=1e-6)
            assert np.all(target[num_boxes:, 4] == -1.0)


def test_max_boxes_is_the_largest_box_count(tmp_path):
    path = write_constant(str(tmp_path / "image.png"), 64, 64, (90, 90, 90))
    lower = np.arange(30, dtype="float32")[:, None]
    many = np.concatenate([lower, lower, lower + 2, lower + 2], axis=1)
    many = np.concatenate([many, np.zeros((30, 1), "float32")], axis=1)
    detections = [build_full_box(64, 64, 0), many]
    validation = build_validation_generator([path] * 2, detections, 2)
    train = build_train_generator(0, [path] * 2, detections, [0, 0], 2)
    assert validation.max_boxes == 30 and train.max_boxes == 30
    assert np.asarray(train[0][1]).shape == (2, 30, 5)
    targets = np.asarray(validation[0][1])
    corners = many[:, :4] / 64.0
    expected = paz.to_numpy(paz.boxes.to_center_form(corners))
    assert np.allclose(targets[1, :, :4], expected, atol=1e-6)
    assert np.all(targets[1, :, 4] == 0.0)
    assert np.all(targets[0, 1:, 4] == -1.0)


def test_validation_keeps_order_and_squashes_boxes(tmp_path):
    paths, _ = write_indexed_images(str(tmp_path), 5)
    detections = []
    for arg in range(5):
        H, W = SIZES[arg % 2]
        box = [W / 8, H / 4, W / 2, H / 2 + arg, arg % 2]
        detections.append(np.array([box], "float32"))
    generator = build_validation_generator(paths, detections, 2)
    images, targets = read_epoch(generator, 0)
    again = read_epoch(generator, 0)
    assert np.array_equal(read_indices(images), np.arange(5))
    assert np.array_equal(images, again[0])
    for arg, target in enumerate(targets):
        H, W = SIZES[arg % 2]
        x_max, y_max = W / 2, H / 2 + arg
        expected = [(W / 8 + x_max) / (2 * W), (H / 4 + y_max) / (2 * H)]
        expected = expected + [(x_max - W / 8) / W, (y_max - H / 4) / H]
        assert np.allclose(target[0], expected + [arg % 2], atol=1e-6)


def test_validation_does_not_flip_or_crop(tmp_path):
    path, detection = write_red_rectangle(str(tmp_path))
    generator = build_validation_generator([path], [detection], 1)
    images, targets = generator[0]
    corners = compute_red_corners(np.asarray(images)[0])
    expected = to_pixel_corners(np.asarray(targets)[0, 0])
    assert np.allclose(expected, [4.0, 8.0, 14.0, 24.0], atol=1e-4)
    assert np.allclose(corners, expected, atol=1.5)


def test_copies_get_one_fixed_augmentation_each(tmp_path):
    rare = (120, 60, 30)
    path_0 = write_constant(str(tmp_path / "rare_0.png"), 40, 56, rare)
    path_1 = write_constant(str(tmp_path / "rare_1.png"), 48, 32, rare)
    path_2 = write_constant(str(tmp_path / "common.png"), 40, 56, (40, 90, 9))
    paths = [path_0, path_1, path_2]
    detections = [build_full_box(40, 56, 1), build_full_box(48, 32, 1)]
    detections.append(build_full_box(40, 56, 0))
    args = (paths, detections, [1], 2)
    paths, detections, copy_indices = oversampling.repeat_rare_images(*args)
    assert copy_indices == [0, 0, 0, 1, 2, 1, 2]
    generator = build_train_generator(5, paths, detections, copy_indices, 7)
    originals = [rare, rare, (40, 90, 9)]
    copies = [compute_copy_color(5, paths[arg], arg) for arg in range(3, 7)]
    assert len(set(copies)) == 4 and rare not in copies
    expected = sorted(tuple(map(float, color)) for color in originals)
    expected = sorted(expected + copies)
    for epoch in range(3):
        colors = read_colors(read_epoch(generator, epoch)[0])
        assert sorted(map(tuple, colors.tolist())) == expected


def test_copies_match_copies_written_offline(tmp_path):
    path = write_texture(str(tmp_path))
    detection = np.array([[5.0, 8.0, 30.0, 33.0, 0.0]], "float32")
    copy_path, copy_boxes = write_offline_copy(3, path, detection, 1)
    args = ([path] * 2, [detection] * 2, [0, 1], 2)
    online = build_train_generator(3, *args)
    args = ([path, copy_path], [detection, copy_boxes], [0, 0], 2)
    offline = build_train_generator(3, *args)
    for epoch in range(3):
        arrays = zip(read_epoch(online, epoch), read_epoch(offline, epoch))
        for online_array, offline_array in arrays:
            assert np.allclose(online_array, offline_array, atol=1e-4)


def test_train_crops_on_the_original_aspect(tmp_path):
    image = np.zeros((8, 64, 3), "uint8")
    image[..., 0] = 4 * np.arange(64)
    image[..., 2] = 100
    path = write_image(str(tmp_path / "wide.png"), image)
    detections = [build_full_box(8, 64, 0)] * 8
    generator = build_train_generator(0, [path] * 8, detections, [0] * 8, 4)
    red = destandardize(read_epoch(generator, 0)[0])[..., 0]
    spans = red.max(axis=(1, 2)) - red.min(axis=(1, 2))
    # an original-aspect crop spans under a fifth of the width
    assert spans.min() < 80


def test_train_boxes_stay_on_their_pixels(tmp_path):
    path, detection = write_red_rectangle(str(tmp_path))
    args = ([path], [detection], [1], 5)
    paths, detections, copy_indices = oversampling.repeat_rare_images(*args)
    generator = build_train_generator(6, paths, detections, copy_indices, 3)
    for epoch in range(3):
        images, targets = read_epoch(generator, epoch)
        for image, target in zip(images, targets):
            corners = compute_red_corners(image)
            if target[0, 4] < 0:
                assert corners is None
            elif np.all(target[0, 2:4] * SIZE >= 3):
                box = to_pixel_corners(target[0])
                assert np.allclose(corners, box, atol=2.0)


def test_validate_labels_accepts_labels_in_range():
    detections = [np.array([[0, 0, 4, 4, 0], [1, 1, 5, 5, 1]], "float32")]
    detections.append(np.zeros((0, 5), "float32"))
    load_generator().validate_labels(detections, 2)


@pytest.mark.parametrize("class_arg", [2.0, -1.0])
def test_validate_labels_rejects_labels_out_of_range(class_arg):
    detections = [np.array([[0, 0, 4, 4, 0]], "float32")]
    detections.append(np.array([[0, 0, 4, 4, class_arg]], "float32"))
    with pytest.raises(ValueError):
        load_generator().validate_labels(detections, 2)
