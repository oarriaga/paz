import os

import numpy as np
import cv2

from paz.datasets import cityscapes


def build_fake_dataset(root, split, city, num_samples):
    image_dir = os.path.join(root, "leftImg8bit", split, city)
    label_dir = os.path.join(root, "gtFine", split, city)
    os.makedirs(image_dir)
    os.makedirs(label_dir)
    for index in range(num_samples):
        prefix = f"{city}_{index:06d}_000000"
        image = np.random.randint(0, 256, (64, 128, 3), "uint8")
        label = np.random.randint(0, 34, (64, 128), "uint8")
        cv2.imwrite(os.path.join(image_dir, prefix + "_leftImg8bit.png"), image)
        cv2.imwrite(os.path.join(label_dir, prefix + "_gtFine_labelIds.png"),
                    label)


def test_id_to_category_mapping():
    table = cityscapes.build_id_to_category()
    assert len(table) == 34
    expected = {0: 0, 7: 1, 11: 2, 17: 3, 21: 4, 23: 5, 24: 6, 26: 7, 33: 7}
    for raw_id, category in expected.items():
        assert table[raw_id] == category


def test_load_pairs_images_and_labels(tmp_path):
    root = str(tmp_path)
    build_fake_dataset(root, "train", "aachen", 3)
    image_paths, label_paths = cityscapes.load(root, "train")
    assert len(image_paths) == len(label_paths) == 3
    for label_path in label_paths:
        assert os.path.exists(label_path)


def test_load_image_and_mask_shapes(tmp_path):
    root = str(tmp_path)
    build_fake_dataset(root, "val", "munich", 1)
    image_paths, label_paths = cityscapes.load(root, "val")
    image = np.asarray(cityscapes.load_image(image_paths[0], (32, 48)))
    mask = np.asarray(cityscapes.load_mask(label_paths[0], (32, 48)))
    assert image.shape == (32, 48, 3)
    assert mask.shape == (32, 48)
    assert mask.min() >= 0 and mask.max() <= 7
