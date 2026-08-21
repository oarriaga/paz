import json
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest

from paz.models.detection.dino_v2_object_detection.detr import RFDETR
from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.datasets.coco import (
    COCOBatchLoader,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_DATASETS_MODULE = (
    "paz.models.detection.dino_v2_object_detection.datasets"
)


def _make_train_config(**overrides):
    defaults = dict(
        dataset_file="coco_json",
        dataset_dir="/fake/path",
        square_resize_div_64=True,
        multi_scale=True,
        expanded_scales=True,
        do_random_resize_via_padding=False,
        batch_size=2,
        grad_accum_steps=3,
        segmentation_head=False,
        num_workers=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def create_mini_coco_dataset(
    root,
    num_images=2,
    num_boxes_per_image=1,
    img_w=64,
    img_h=48,
    all_iscrowd=False,
    include_test=False,
):
    splits = [("train", num_images), ("valid", 1)]
    if include_test:
        splits.append(("test", 1))

    for split_dir, count in splits:
        d = root / split_dir
        d.mkdir(parents=True, exist_ok=True)

        images_meta = []
        annotations = []
        ann_id = 1
        for i in range(1, count + 1):
            fname = f"img_{i:04d}.png"
            img = PIL.Image.fromarray(
                np.random.randint(0, 256, (img_h, img_w, 3), dtype=np.uint8),
                "RGB",
            )
            img.save(str(d / fname))

            images_meta.append(
                {"id": i, "file_name": fname, "width": img_w, "height": img_h}
            )
            for b in range(num_boxes_per_image):
                bx = 5 + b * 10
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": i,
                        "category_id": 1,
                        "bbox": [bx, 5, 10, 10],  # xywh
                        "area": 100.0,
                        "iscrowd": 1 if all_iscrowd else 0,
                    }
                )
                ann_id += 1

        coco_json = {
            "images": images_meta,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "fish"}],
        }
        with open(d / "_annotations.coco.json", "w") as f:
            json.dump(coco_json, f)


def create_no_annotation_dataset(root, num_images=2):
    for split_dir, count in [("train", num_images), ("valid", 1)]:
        d = root / split_dir
        d.mkdir(parents=True, exist_ok=True)

        images_meta = []
        for i in range(1, count + 1):
            fname = f"img_{i:04d}.png"
            img = PIL.Image.fromarray(
                np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8), "RGB"
            )
            img.save(str(d / fname))
            images_meta.append(
                {"id": i, "file_name": fname, "width": 64, "height": 48}
            )

        coco_json = {
            "images": images_meta,
            "annotations": [],
            "categories": [{"id": 1, "name": "fish"}],
        }
        with open(d / "_annotations.coco.json", "w") as f:
            json.dump(coco_json, f)


# =========================================================================
# Unit tests (patched build_dataset)
# =========================================================================


class TestBuildDataLoaderUnit:

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_returns_none_when_dataset_dir_empty(self, mock_build):
        config = _make_train_config(dataset_dir="")
        result = RFDETR.build_data_loader(config, "train", {})
        assert result is None
        mock_build.assert_not_called()

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_returns_none_when_dataset_dir_none(self, mock_build):
        config = _make_train_config(dataset_dir=None)
        result = RFDETR.build_data_loader(config, "train", {})
        assert result is None
        mock_build.assert_not_called()

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_returns_none_on_assertion_error(self, mock_build):
        mock_build.side_effect = AssertionError("path does not exist")
        config = _make_train_config(dataset_dir="/nonexistent")
        result = RFDETR.build_data_loader(config, "train", {})
        assert result is None

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_returns_none_on_file_not_found(self, mock_build):
        mock_build.side_effect = FileNotFoundError("no annotation file")
        config = _make_train_config(dataset_dir="/nonexistent")
        result = RFDETR.build_data_loader(config, "train", {})
        assert result is None

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_args_namespace_train(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_build.return_value = mock_dataset

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir="/data/coco",
            square_resize_div_64=True,
            multi_scale=True,
            expanded_scales=False,
            do_random_resize_via_padding=True,
            segmentation_head=True,
        )
        kwargs = {"patch_size": 16, "num_windows": 2, "resolution": 512}
        loader = RFDETR.build_data_loader(config, "train", kwargs)

        # Inspect the _Args that was passed to build_dataset
        args = mock_build.call_args[0][1]
        assert args.dataset_file == "roboflow"
        assert args.dataset_dir == "/data/coco"
        assert args.square_resize_div_64 is True
        assert args.multi_scale is True  # train → uses config value
        assert args.expanded_scales is False
        assert args.do_random_resize_via_padding is True
        assert args.patch_size == 16
        assert args.num_windows == 2
        assert args.segmentation_head is True

        # Verify split and resolution
        assert mock_build.call_args[0][0] == "train"
        assert mock_build.call_args[0][2] == 512

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_args_namespace_val_forces_multi_scale_false(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config(multi_scale=True)
        RFDETR.build_data_loader(config, "val", {})

        args = mock_build.call_args[0][1]
        assert args.multi_scale is False

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_default_patch_size_and_num_windows(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config()
        RFDETR.build_data_loader(config, "train", {})

        args = mock_build.call_args[0][1]
        assert args.patch_size == 14
        assert args.num_windows == 4

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_default_resolution(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config()
        RFDETR.build_data_loader(config, "train", {})

        assert mock_build.call_args[0][2] == 560

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_batch_size_is_product(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_build.return_value = mock_dataset

        config = _make_train_config(batch_size=4, grad_accum_steps=8)
        loader = RFDETR.build_data_loader(config, "train", {})

        assert isinstance(loader, COCOBatchLoader)
        assert loader.batch_size == 32

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_train_shuffle_and_drop_last(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_build.return_value = mock_dataset

        config = _make_train_config()
        loader = RFDETR.build_data_loader(config, "train", {})

        assert loader.shuffle is True
        assert loader.drop_last is True

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_val_no_shuffle_no_drop(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config()
        loader = RFDETR.build_data_loader(config, "val", {})

        assert loader.shuffle is False
        assert loader.drop_last is False

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_multi_scale_false_config_stays_false_for_train(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config(multi_scale=False)
        RFDETR.build_data_loader(config, "train", {})

        args = mock_build.call_args[0][1]
        assert args.multi_scale is False

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_segmentation_head_defaults_false(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config(segmentation_head=False)
        RFDETR.build_data_loader(config, "train", {})

        args = mock_build.call_args[0][1]
        assert args.segmentation_head is False

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_coco_dataset_file(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config(dataset_file="coco")
        RFDETR.build_data_loader(config, "train", {})

        args = mock_build.call_args[0][1]
        assert args.dataset_file == "coco"

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_grad_accum_steps_one(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config(batch_size=7, grad_accum_steps=1)
        loader = RFDETR.build_data_loader(config, "train", {})

        assert loader.batch_size == 7

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_square_resize_div_64_false(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config(square_resize_div_64=False)
        RFDETR.build_data_loader(config, "train", {})

        args = mock_build.call_args[0][1]
        assert args.square_resize_div_64 is False

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_do_random_resize_via_padding_propagates(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        for flag in [True, False]:
            config = _make_train_config(do_random_resize_via_padding=flag)
            RFDETR.build_data_loader(config, "train", {})
            args = mock_build.call_args[0][1]
            assert args.do_random_resize_via_padding is flag

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_kwargs_override_defaults(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config()
        RFDETR.build_data_loader(
            config, "train",
            {"patch_size": 20, "num_windows": 1, "resolution": 700},
        )

        args = mock_build.call_args[0][1]
        assert args.patch_size == 20
        assert args.num_windows == 1
        assert mock_build.call_args[0][2] == 700

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_split_passed_to_build_dataset(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        config = _make_train_config()

        for split in ["train", "val"]:
            RFDETR.build_data_loader(config, split, {})
            assert mock_build.call_args[0][0] == split

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_expanded_scales_propagates(self, mock_build):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=5)
        mock_build.return_value = mock_dataset

        for flag in [True, False]:
            config = _make_train_config(expanded_scales=flag)
            RFDETR.build_data_loader(config, "train", {})
            args = mock_build.call_args[0][1]
            assert args.expanded_scales is flag

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    def test_dataset_dir_whitespace_is_truthy(self, mock_build):
        mock_build.side_effect = AssertionError("path does not exist")
        config = _make_train_config(dataset_dir="   ")
        result = RFDETR.build_data_loader(config, "train", {})
        # Should call build_dataset (whitespace is truthy) then catch error
        mock_build.assert_called_once()
        assert result is None

    @patch(f"{_DATASETS_MODULE}.build_dataset")
    @pytest.mark.parametrize(
        "bs,accum,expected",
        [(1, 1, 1), (2, 4, 8), (8, 2, 16), (1, 16, 16), (16, 1, 16)],
    )
    def test_batch_size_product_parametrized(
        self, mock_build, bs, accum, expected
    ):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_build.return_value = mock_dataset

        config = _make_train_config(batch_size=bs, grad_accum_steps=accum)
        loader = RFDETR.build_data_loader(config, "train", {})
        assert loader.batch_size == expected


# =========================================================================
# Integration test (real mini dataset on disk)
# =========================================================================


class TestBuildDataLoaderIntegration:

    def test_train_loader_yields_valid_batches(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=2)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=2,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "train", {"resolution": 560, "patch_size": 14, "num_windows": 4}  # fmt: skip
        )
        assert loader is not None
        assert isinstance(loader, COCOBatchLoader)

        for images_np, targets in loader:
            # images_np: (B, H, W, 3) float32
            assert isinstance(images_np, np.ndarray)
            assert images_np.dtype == np.float32
            assert images_np.ndim == 4
            assert images_np.shape[0] <= 2
            assert images_np.shape[3] == 3

            # Each target is a dict with numpy arrays
            for tgt in targets:
                assert "boxes" in tgt
                assert "labels" in tgt
                assert tgt["boxes"].ndim == 2
                assert tgt["boxes"].shape[1] == 4
                assert tgt["labels"].ndim == 1
            break  # one batch is enough

    def test_val_loader_yields_valid_batches(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=2)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        assert loader is not None

        for images_np, targets in loader:
            assert images_np.ndim == 4
            assert images_np.shape[0] == 1
            assert len(targets) == 1
            break

    def test_nonexistent_dir_returns_none(self, tmp_path):
        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path / "does_not_exist"),
        )
        loader = RFDETR.build_data_loader(config, "train", {})
        assert loader is None

    def test_train_normalized_pixel_range(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "train", {"resolution": 560}
        )
        for images_np, targets in loader:
            # After ImageNet normalization some values will be negative
            # (mean subtraction) and some > 1 (divided by std < 1)
            assert images_np.min() < 0.0 or images_np.max() > 1.0
            break

    def test_boxes_are_normalized_cxcywh(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for _, targets in loader:
            boxes = targets[0]["boxes"]
            assert boxes.shape[1] == 4
            # Normalized coords should all be in [0, 1]
            assert np.all(boxes >= 0.0), f"Negative box coords: {boxes}"
            assert np.all(boxes <= 1.0), f"Box coords > 1: {boxes}"
            break

    def test_loader_length(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=4)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=2,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "train", {"resolution": 560}
        )
        # 4 < effective_bs 2 * min_batches 5 → oversampled to 10 samples;
        # 10 / batch_size 2, drop_last=True → 5 batches
        assert len(loader) == 5

    def test_val_loader_no_shuffle_deterministic(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=2)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=2,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )

        batches_1 = [(img.copy(), [t.copy() for t in tgt]) for img, tgt in loader]  # fmt: skip
        loader2 = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        batches_2 = [(img.copy(), [t.copy() for t in tgt]) for img, tgt in loader2]  # fmt: skip

        assert len(batches_1) == len(batches_2)
        for (img1, _), (img2, _) in zip(batches_1, batches_2):
            np.testing.assert_array_equal(img1, img2)

    def test_square_resize_produces_square_images(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1, img_w=100, img_h=60)

        resolution = 560
        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": resolution}
        )
        for images_np, _ in loader:
            assert images_np.shape[1] == resolution  # H
            assert images_np.shape[2] == resolution  # W
            break

    @pytest.mark.parametrize("resolution", [384, 512, 560])
    def test_different_resolutions(self, tmp_path, resolution):
        create_mini_coco_dataset(tmp_path, num_images=1)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": resolution}
        )
        for images_np, _ in loader:
            assert images_np.shape[1] == resolution
            assert images_np.shape[2] == resolution
            break

    def test_batch_size_larger_than_dataset_train_drops(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=2)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=10,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "train", {"resolution": 560}
        )
        assert loader is not None
        # 2 < effective_bs 10 * min_batches 5 → oversampled to 50 samples;
        # 50 / batch_size 10, drop_last=True → 5 batches
        assert len(loader) == 5
        batches = list(loader)
        assert len(batches) == 5

    def test_batch_size_larger_than_dataset_val_keeps(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=2)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=10,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        # valid/ has 1 image, batch_size=10, drop_last=False → 1 batch
        assert len(loader) == 1
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0].shape[0] == 1  # only 1 image

    def test_multiple_boxes_per_image(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1, num_boxes_per_image=5)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for _, targets in loader:
            # 5 boxes per image
            assert targets[0]["boxes"].shape[0] == 5
            assert targets[0]["labels"].shape[0] == 5
            break

    def test_zero_annotations_yields_empty_boxes(self, tmp_path):
        create_no_annotation_dataset(tmp_path, num_images=1)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for _, targets in loader:
            assert targets[0]["boxes"].shape == (0, 4)
            assert targets[0]["labels"].shape == (0,)
            break

    def test_all_iscrowd_yields_empty_boxes(self, tmp_path):
        create_mini_coco_dataset(
            tmp_path, num_images=1, num_boxes_per_image=3, all_iscrowd=True
        )

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for _, targets in loader:
            assert targets[0]["boxes"].shape == (0, 4)
            break

    def test_target_contains_all_expected_keys(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        expected_keys = {"boxes", "labels", "image_id", "area", "iscrowd",
                         "orig_size", "size"}
        for _, targets in loader:
            assert expected_keys.issubset(set(targets[0].keys()))
            break

    def test_image_dtype_and_channels(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        for split in ["train", "val"]:
            loader = RFDETR.build_data_loader(
                config, split, {"resolution": 560}
            )
            if loader is None:
                continue
            for images_np, _ in loader:
                assert images_np.dtype == np.float32
                assert images_np.shape[-1] == 3
                break

    def test_full_iteration_no_crash(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=4)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=2,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "train", {"resolution": 560}
        )
        count = 0
        for images_np, targets in loader:
            count += 1
            assert images_np.shape[0] == 2
            assert len(targets) == 2
        assert count == 5  # 4 < 10 → oversampled to 10 / bs=2 → 5 batches

    def test_grad_accum_increases_effective_batch(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=6)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=2,
            grad_accum_steps=3,  # effective = 6
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "train", {"resolution": 560}
        )
        assert loader.batch_size == 6
        # 6 < effective_bs 6 * min_batches 5 → oversampled to 30 samples;
        # 30 / batch_size 6, drop_last=True → 5 batches
        assert len(loader) == 5

    def test_wide_image(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1, img_w=1000, img_h=100)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for images_np, targets in loader:
            assert images_np.shape[1] == 560
            assert images_np.shape[2] == 560
            assert targets[0]["boxes"].shape[1] == 4
            break

    def test_tall_image(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1, img_w=100, img_h=1000)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for images_np, targets in loader:
            assert images_np.shape[1] == 560
            assert images_np.shape[2] == 560
            break

    def test_square_image(self, tmp_path):
        create_mini_coco_dataset(tmp_path, num_images=1, img_w=100, img_h=100)

        config = _make_train_config(
            dataset_file="roboflow",
            dataset_dir=str(tmp_path),
            batch_size=1,
            grad_accum_steps=1,
            multi_scale=False,
            square_resize_div_64=True,
        )
        loader = RFDETR.build_data_loader(
            config, "val", {"resolution": 560}
        )
        for images_np, _ in loader:
            assert images_np.shape[1] == 560
            assert images_np.shape[2] == 560
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
