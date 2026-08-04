import random
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

# ---- Ensure both packages are importable --------------------------------

# Keras (numpy/PIL) transforms & coco utilities
from paz.models.detection.dino_v2_object_detection.datasets import (
    transforms as K,
)
from paz.models.detection.dino_v2_object_detection.datasets.coco import (
    compute_multi_scale_scales,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
    convert_coco as K_ConvertCoco,
)

# PyTorch reference transforms
_PT_ROOT = str(
    Path(__file__).resolve().parents[5]
    / "examples"
    / "rf-detr_original_pytorch_implementation"
)
if not Path(_PT_ROOT).is_dir():
    pytest.skip("RF-DETR reference unavailable", allow_module_level=True)
pytest.importorskip("torch")
if _PT_ROOT not in sys.path:
    sys.path.insert(0, _PT_ROOT)

import torch
import rfdetr.datasets.transforms as PT
from rfdetr.datasets.coco import (
    compute_multi_scale_scales as pt_compute_multi_scale_scales,
    make_coco_transforms as pt_make_coco_transforms,
    make_coco_transforms_square_div_64 as pt_make_coco_transforms_square_div_64,
    ConvertCoco as PT_ConvertCoco,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 12345


def _make_pil_image(w=120, h=80):
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return PIL.Image.fromarray(arr, "RGB")


def _make_target_np(n_boxes=3, w=120, h=80):
    rng = np.random.RandomState(7)
    x1 = rng.uniform(0, w * 0.6, n_boxes).astype(np.float32)
    y1 = rng.uniform(0, h * 0.6, n_boxes).astype(np.float32)
    x2 = x1 + rng.uniform(10, w * 0.3, n_boxes).astype(np.float32)
    y2 = y1 + rng.uniform(10, h * 0.3, n_boxes).astype(np.float32)
    x2 = np.clip(x2, 0, w).astype(np.float32)
    y2 = np.clip(y2, 0, h).astype(np.float32)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    area = (x2 - x1) * (y2 - y1)
    return {
        "boxes": boxes,
        "labels": np.arange(1, n_boxes + 1, dtype=np.int64),
        "area": area,
        "iscrowd": np.zeros(n_boxes, dtype=np.int64),
        "image_id": np.array([1], dtype=np.int64),
        "size": np.array([h, w], dtype=np.int64),
        "orig_size": np.array([h, w], dtype=np.int64),
    }


def _make_target_pt(np_target):
    return {k: torch.from_numpy(v.copy()) for k, v in np_target.items()}


def _copy_np_target(tgt):
    return {k: v.copy() for k, v in tgt.items()}


def _compare_images_pil(img_k, img_pt):
    arr_k = np.asarray(img_k)
    arr_pt = np.asarray(img_pt)
    np.testing.assert_array_equal(arr_k, arr_pt)


def _compare_images_tensor(img_k, img_pt, atol=1e-5):
    pt_np = img_pt.permute(1, 2, 0).numpy()
    np.testing.assert_allclose(img_k, pt_np, atol=atol, rtol=0)


def _compare_target(tgt_k, tgt_pt, atol=1e-5):
    for key in ("boxes", "labels", "area"):
        if key not in tgt_k:
            continue
        val_k = tgt_k[key]
        val_pt = tgt_pt[key].numpy()
        np.testing.assert_allclose(
            val_k,
            val_pt,
            atol=atol,
            rtol=0,
            err_msg=f"Mismatch in target['{key}']",
        )


# =========================================================================
# Tests: standalone functions
# =========================================================================


class TestHFlip:

    def test_deterministic(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.hflip(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.hflip(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_double_flip_is_identity(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()

        img2, tgt2 = K.hflip(img, _copy_np_target(tgt_np))
        img3, tgt3 = K.hflip(img2, _copy_np_target(tgt2))

        _compare_images_pil(img, img3)
        np.testing.assert_allclose(tgt_np["boxes"], tgt3["boxes"], atol=1e-5)

    def test_empty_boxes(self):
        img = _make_pil_image()
        tgt = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "area": np.zeros(0, dtype=np.float32),
            "iscrowd": np.zeros(0, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([80, 120], dtype=np.int64),
            "orig_size": np.array([80, 120], dtype=np.int64),
        }
        tgt_pt = _make_target_pt(tgt)

        img_k, tgt_k = K.hflip(img, _copy_np_target(tgt))
        img_pt, tgt_pt2 = PT.hflip(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        assert tgt_k["boxes"].shape == (0, 4)

    def test_single_box_at_edge(self):
        w, h = 100, 80
        img = _make_pil_image(w, h)
        tgt = {
            "boxes": np.array([[0, 10, w, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
            "area": np.array([w * 40], dtype=np.float32),
            "iscrowd": np.zeros(1, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        tgt_pt = _make_target_pt(tgt)

        _, tgt_k = K.hflip(img, _copy_np_target(tgt))
        _, tgt_pt2 = PT.hflip(img, tgt_pt)

        _compare_target(tgt_k, tgt_pt2)
        # Full-width box should be identical after flip
        np.testing.assert_allclose(
            tgt_k["boxes"], tgt["boxes"], atol=1e-5
        )


class TestCrop:

    def test_deterministic(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()
        tgt_pt = _make_target_pt(tgt_np)
        region = (5, 10, 50, 60)  # top, left, h, w

        img_k, tgt_k = K.crop(img, _copy_np_target(tgt_np), region)
        img_pt, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_full_image_crop_is_identity(self):
        w, h = 120, 80
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(3, w, h)
        tgt_pt = _make_target_pt(tgt_np)
        region = (0, 0, h, w)

        img_k, tgt_k = K.crop(img, _copy_np_target(tgt_np), region)
        img_pt, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_images_pil(img_k, img_pt)
        _compare_images_pil(img, img_k)
        _compare_target(tgt_k, tgt_pt2)

    def test_crop_removes_out_of_bounds_boxes(self):
        w, h = 200, 200
        img = _make_pil_image(w, h)
        # box in top-left quadrant, box in bottom-right quadrant
        tgt = {
            "boxes": np.array(
                [[10, 10, 40, 40], [150, 150, 190, 190]], dtype=np.float32
            ),
            "labels": np.array([1, 2], dtype=np.int64),
            "area": np.array([900, 1600], dtype=np.float32),
            "iscrowd": np.zeros(2, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        tgt_pt = _make_target_pt(tgt)
        # Crop only top-left 100x100 → second box is outside
        region = (0, 0, 100, 100)

        _, tgt_k = K.crop(img, _copy_np_target(tgt), region)
        _, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_target(tgt_k, tgt_pt2)
        assert tgt_k["boxes"].shape[0] == tgt_pt2["boxes"].shape[0]
        # First box should survive, second should be clipped to zero area
        assert tgt_k["boxes"].shape[0] == 1

    def test_crop_clips_partially_visible_box(self):
        w, h = 200, 200
        img = _make_pil_image(w, h)
        # Box from (80,80) to (130,130)  -- partially in a (0,0,100,100) crop
        tgt = {
            "boxes": np.array([[80, 80, 130, 130]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
            "area": np.array([2500], dtype=np.float32),
            "iscrowd": np.zeros(1, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        tgt_pt = _make_target_pt(tgt)
        region = (0, 0, 100, 100)

        _, tgt_k = K.crop(img, _copy_np_target(tgt), region)
        _, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_target(tgt_k, tgt_pt2)
        assert tgt_k["boxes"].shape[0] == 1
        # Clipped box should be (80,80,100,100)
        np.testing.assert_allclose(
            tgt_k["boxes"][0],
            np.array([80, 80, 100, 100], dtype=np.float32),
            atol=1e-5,
        )

    def test_crop_empty_boxes(self):
        img = _make_pil_image()
        tgt = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "area": np.zeros(0, dtype=np.float32),
            "iscrowd": np.zeros(0, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([80, 120], dtype=np.int64),
            "orig_size": np.array([80, 120], dtype=np.int64),
        }
        tgt_pt = _make_target_pt(tgt)
        region = (5, 5, 40, 40)

        img_k, tgt_k = K.crop(img, _copy_np_target(tgt), region)
        img_pt, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_images_pil(img_k, img_pt)
        assert tgt_k["boxes"].shape == (0, 4)

    def test_various_crop_regions(self):
        w, h = 200, 150
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(5, w, h)
        regions = [
            (0, 0, 50, 50),
            (0, 0, h, w),       # full
            (10, 10, 100, 100),
            (50, 50, 100, 150),
            (0, 100, 150, 100),
        ]
        for region in regions:
            tgt_pt = _make_target_pt(tgt_np)
            img_k, tgt_k = K.crop(img, _copy_np_target(tgt_np), region)
            img_pt, tgt_pt2 = PT.crop(img, tgt_pt, region)
            _compare_images_pil(img_k, img_pt)
            _compare_target(tgt_k, tgt_pt2)


class TestResize:

    def test_single_size(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.resize(img, _copy_np_target(tgt_np), 100, max_size=1333)  # fmt: skip
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, 100, max_size=1333)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_tuple_size(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.resize(img, _copy_np_target(tgt_np), (300, 400))
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, (300, 400))

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_resize_identity(self):
        w, h = 120, 80
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(3, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        # Tuple arg is (w, h); _get_size reverses to (h, w) internally
        img_k, tgt_k = K.resize(img, _copy_np_target(tgt_np), (w, h))
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, (w, h))

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)
        # Boxes should be unchanged
        np.testing.assert_allclose(tgt_np["boxes"], tgt_k["boxes"], atol=1e-5)

    def test_max_size_limiting(self):
        img = _make_pil_image(1000, 500)
        tgt_np = _make_target_np(2, 1000, 500)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.resize(img, _copy_np_target(tgt_np), 800, max_size=900)
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, 800, max_size=900)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)
        # Long edge should not exceed max_size
        w_out, h_out = img_k.size
        assert max(w_out, h_out) <= 900

    def test_none_target(self):
        img = _make_pil_image(200, 150)
        img_k, tgt_k = K.resize(img, None, 100)
        img_pt, tgt_pt = PT.resize(img, None, 100)
        _compare_images_pil(img_k, img_pt)
        assert tgt_k is None
        assert tgt_pt is None

    def test_empty_boxes(self):
        img = _make_pil_image(200, 150)
        tgt = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "area": np.zeros(0, dtype=np.float32),
            "iscrowd": np.zeros(0, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([150, 200], dtype=np.int64),
            "orig_size": np.array([150, 200], dtype=np.int64),
        }
        tgt_pt = _make_target_pt(tgt)
        img_k, tgt_k = K.resize(img, _copy_np_target(tgt), 100)
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, 100)
        _compare_images_pil(img_k, img_pt)
        assert tgt_k["boxes"].shape == (0, 4)

    @pytest.mark.parametrize("size", [50, 100, 200, 400])
    def test_various_sizes(self, size):
        img = _make_pil_image(300, 200)
        tgt_np = _make_target_np(3, 300, 200)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.resize(img, _copy_np_target(tgt_np), size, max_size=1333)  # fmt: skip
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, size, max_size=1333)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)


class TestPad:

    def test_bottom_right(self):
        img = _make_pil_image(100, 80)
        tgt_np = _make_target_np(3, 100, 80)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.pad(img, _copy_np_target(tgt_np), (20, 30))
        img_pt, tgt_pt2 = PT.pad(img, tgt_pt, (20, 30))

        _compare_images_pil(img_k, img_pt)
        np.testing.assert_array_equal(tgt_k["size"], tgt_pt2["size"].numpy())

    def test_zero_padding_is_identity(self):
        img = _make_pil_image(100, 80)
        tgt_np = _make_target_np(3, 100, 80)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.pad(img, _copy_np_target(tgt_np), (0, 0))
        img_pt, tgt_pt2 = PT.pad(img, tgt_pt, (0, 0))

        _compare_images_pil(img_k, img_pt)
        _compare_images_pil(img, img_k)

    def test_none_target(self):
        img = _make_pil_image(100, 80)
        img_k, tgt_k = K.pad(img, None, (10, 10))
        img_pt, tgt_pt = PT.pad(img, None, (10, 10))
        _compare_images_pil(img_k, img_pt)
        assert tgt_k is None
        assert tgt_pt is None

    def test_padded_region_is_black(self):
        w, h = 50, 40
        img = PIL.Image.new("RGB", (w, h), color=(255, 128, 64))
        tgt = _make_target_np(1, w, h)
        img_k, _ = K.pad(img, _copy_np_target(tgt), (10, 20))
        arr = np.asarray(img_k)
        # Bottom 20 rows should be black
        np.testing.assert_array_equal(arr[h:, :, :], 0)
        # Right 10 cols should be black
        np.testing.assert_array_equal(arr[:, w:, :], 0)

    @pytest.mark.parametrize(
        "padding", [(0, 0), (1, 1), (10, 0), (0, 10), (50, 50)]
    )
    def test_various_paddings(self, padding):
        img = _make_pil_image(100, 80)
        tgt_np = _make_target_np(2, 100, 80)
        tgt_pt = _make_target_pt(tgt_np)
        img_k, tgt_k = K.pad(img, _copy_np_target(tgt_np), padding)
        img_pt, tgt_pt2 = PT.pad(img, tgt_pt, padding)
        _compare_images_pil(img_k, img_pt)
        np.testing.assert_array_equal(tgt_k["size"], tgt_pt2["size"].numpy())


# =========================================================================
# Tests: transform classes
# =========================================================================


class TestRandomHorizontalFlip:

    def test_with_seed(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()
        tgt_pt = _make_target_pt(tgt_np)

        random.seed(SEED)
        img_k, tgt_k = K.random_horizontal_flip(p=0.5)(img, _copy_np_target(tgt_np))  # fmt: skip

        random.seed(SEED)
        img_pt, tgt_pt2 = PT.RandomHorizontalFlip(p=0.5)(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_p_zero_never_flips(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()

        for _ in range(20):
            out_img, out_tgt = K.random_horizontal_flip(p=0.0)(
                img, _copy_np_target(tgt_np)
            )
            _compare_images_pil(img, out_img)
            np.testing.assert_array_equal(tgt_np["boxes"], out_tgt["boxes"])

    def test_p_one_always_flips(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.random_horizontal_flip(p=1.0)(
            img, _copy_np_target(tgt_np)
        )
        img_ref, tgt_ref = K.hflip(img, _copy_np_target(tgt_np))

        _compare_images_pil(img_k, img_ref)
        np.testing.assert_allclose(tgt_k["boxes"], tgt_ref["boxes"], atol=1e-6)


class TestRandomResize:

    def test_with_seed(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        random.seed(SEED)
        img_k, tgt_k = K.random_resize([400, 500, 600])(
            img, _copy_np_target(tgt_np)
        )

        random.seed(SEED)
        img_pt, tgt_pt2 = PT.RandomResize([400, 500, 600])(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_single_size_list(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.random_resize([300])(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.RandomResize([300])(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_with_max_size(self):
        img = _make_pil_image(800, 400)
        tgt_np = _make_target_np(2, 800, 400)
        tgt_pt = _make_target_pt(tgt_np)

        random.seed(SEED)
        img_k, tgt_k = K.random_resize([600, 700], max_size=500)(
            img, _copy_np_target(tgt_np)
        )

        random.seed(SEED)
        img_pt, tgt_pt2 = PT.RandomResize([600, 700], max_size=500)(
            img, tgt_pt
        )

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)


class TestSquareResize:

    def test_deterministic(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        random.seed(SEED)
        img_k, tgt_k = K.square_resize([560])(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = PT.SquareResize([560])(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_already_square(self):
        img = _make_pil_image(100, 100)
        tgt_np = _make_target_np(2, 100, 100)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.square_resize([100])(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.SquareResize([100])(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_none_target(self):
        img = _make_pil_image(200, 150)
        img_k, tgt_k = K.square_resize([300])(img, None)
        img_pt, tgt_pt = PT.SquareResize([300])(img, None)
        _compare_images_pil(img_k, img_pt)
        assert tgt_k is None

    @pytest.mark.parametrize("size", [64, 128, 256, 512])
    def test_various_sizes(self, size):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.square_resize([size])(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.SquareResize([size])(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)
        assert img_k.size == (size, size)


class TestRandomSizeCrop:

    def test_crop_logic_parity(self):
        img = _make_pil_image(500, 400)
        tgt_np = _make_target_np(3, 500, 400)
        tgt_pt = _make_target_pt(tgt_np)

        region = (10, 20, 384, 400)
        img_k, tgt_k = K.crop(img, _copy_np_target(tgt_np), region)
        img_pt, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_output_in_range(self):
        img = _make_pil_image(500, 400)
        tgt_np = _make_target_np(3, 500, 400)

        for _ in range(50):
            rc = K.random_size_crop(384, 600)
            out_img, _ = rc(img, _copy_np_target(tgt_np))
            w_out, h_out = out_img.size
            assert 384 <= h_out <= 400, f"h={h_out} out of range"
            assert 384 <= w_out <= 500, f"w={w_out} out of range"

    def test_min_equals_max(self):
        img = _make_pil_image(200, 200)
        tgt_np = _make_target_np(2, 200, 200)

        for _ in range(10):
            rc = K.random_size_crop(100, 100)
            out_img, _ = rc(img, _copy_np_target(tgt_np))
            assert out_img.size == (100, 100)

    def test_large_max_clamps_to_image(self):
        w, h = 150, 120
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(2, w, h)

        for _ in range(20):
            rc = K.random_size_crop(50, 9999)
            out_img, _ = rc(img, _copy_np_target(tgt_np))
            wo, ho = out_img.size
            assert wo <= w and ho <= h

    def test_preserves_valid_boxes(self):
        w, h = 300, 300
        img = _make_pil_image(w, h)
        tgt = {
            "boxes": np.array([[50, 50, 100, 100]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
            "area": np.array([2500], dtype=np.float32),
            "iscrowd": np.zeros(1, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        # Crop that fully contains the box
        region = (0, 0, 200, 200)
        _, tgt_k = K.crop(img, _copy_np_target(tgt), region)
        assert tgt_k["boxes"].shape[0] == 1
        np.testing.assert_allclose(
            tgt_k["boxes"][0],
            np.array([50, 50, 100, 100], dtype=np.float32),
            atol=1e-5,
        )


class TestRandomSelect:

    def test_both_branches(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()
        tgt_pt = _make_target_pt(tgt_np)

        t1_k = K.random_horizontal_flip(p=1.0)
        t2_k = K.random_horizontal_flip(p=0.0)
        t1_pt = PT.RandomHorizontalFlip(p=1.0)
        t2_pt = PT.RandomHorizontalFlip(p=0.0)

        random.seed(0)
        img_k, tgt_k = K.random_select(t1_k, t2_k, p=0.5)(
            img, _copy_np_target(tgt_np)
        )

        random.seed(0)
        img_pt, tgt_pt2 = PT.RandomSelect(t1_pt, t2_pt, p=0.5)(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_p_zero_always_second(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()

        flip = K.random_horizontal_flip(p=1.0)
        noop = K.random_horizontal_flip(p=0.0)

        for _ in range(20):
            out_img, _ = K.random_select(flip, noop, p=0.0)(
                img, _copy_np_target(tgt_np)
            )
            _compare_images_pil(img, out_img)  # noop => unchanged

    def test_p_one_always_first(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()

        flip = K.random_horizontal_flip(p=1.0)
        noop = K.random_horizontal_flip(p=0.0)

        ref_img, _ = K.hflip(img, _copy_np_target(tgt_np))
        for _ in range(20):
            out_img, _ = K.random_select(flip, noop, p=1.0)(
                img, _copy_np_target(tgt_np)
            )
            _compare_images_pil(ref_img, out_img)


class TestCenterCrop:

    def test_deterministic(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.center_crop((100, 120))(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.CenterCrop((100, 120))(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_full_size(self):
        w, h = 120, 80
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(3, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.center_crop((h, w))(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.CenterCrop((h, w))(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_images_pil(img, img_k)
        _compare_target(tgt_k, tgt_pt2)


class TestRandomPad:

    def test_with_seed(self):
        img = _make_pil_image(100, 80)
        tgt_np = _make_target_np(3, 100, 80)
        tgt_pt = _make_target_pt(tgt_np)

        random.seed(SEED)
        img_k, tgt_k = K.random_pad(20)(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = PT.RandomPad(20)(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        np.testing.assert_array_equal(tgt_k["size"], tgt_pt2["size"].numpy())

    def test_max_pad_zero(self):
        img = _make_pil_image(100, 80)
        tgt_np = _make_target_np(3, 100, 80)

        img_k, _ = K.random_pad(0)(img, _copy_np_target(tgt_np))
        _compare_images_pil(img, img_k)


class TestRandomCrop:

    def test_output_size(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)

        for _ in range(20):
            rc = K.random_crop((80, 80))
            out_img, _ = rc(img, _copy_np_target(tgt_np))
            assert out_img.size == (80, 80)

    def test_single_int_size(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)

        rc = K.random_crop(50)
        out_img, _ = rc(img, _copy_np_target(tgt_np))
        assert out_img.size == (50, 50)


# =========================================================================
# Tests: ToTensor & Normalize
# =========================================================================


class TestToTensor:

    def test_values(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()
        tgt_pt = _make_target_pt(tgt_np)

        arr_k, _ = K.to_tensor()(img, _copy_np_target(tgt_np))
        arr_pt, _ = PT.ToTensor()(img, tgt_pt)

        # PyTorch: (C,H,W); Keras: (H,W,C)
        pt_np = arr_pt.permute(1, 2, 0).numpy()
        np.testing.assert_allclose(arr_k, pt_np, atol=1e-6)

    def test_range_zero_one(self):
        img = _make_pil_image()
        arr_k, _ = K.to_tensor()(img, _make_target_np())
        assert arr_k.min() >= 0.0
        assert arr_k.max() <= 1.0

    def test_dtype_float32(self):
        img = _make_pil_image()
        arr_k, _ = K.to_tensor()(img, _make_target_np())
        assert arr_k.dtype == np.float32

    def test_shape(self):
        w, h = 120, 80
        img = _make_pil_image(w, h)
        arr_k, _ = K.to_tensor()(img, _make_target_np())
        assert arr_k.shape == (h, w, 3)

    def test_pure_white(self):
        img = PIL.Image.new("RGB", (10, 10), color=(255, 255, 255))
        arr, _ = K.to_tensor()(img, _make_target_np(1, 10, 10))
        np.testing.assert_allclose(arr, 1.0, atol=1e-6)

    def test_pure_black(self):
        img = PIL.Image.new("RGB", (10, 10), color=(0, 0, 0))
        arr, _ = K.to_tensor()(img, _make_target_np(1, 10, 10))
        np.testing.assert_allclose(arr, 0.0, atol=1e-6)


class TestNormalize:

    def test_values_and_boxes(self):
        img = _make_pil_image(100, 80)
        tgt_np = _make_target_np(3, 100, 80)
        tgt_pt = _make_target_pt(tgt_np)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Keras: ToTensor → (H,W,C) then Normalize
        arr_k, _ = K.to_tensor()(img, _copy_np_target(tgt_np))
        arr_k, tgt_k = K.normalize(mean, std)(arr_k, _copy_np_target(tgt_np))

        # PyTorch: F.to_tensor → (C,H,W) then Normalize
        arr_pt, _ = PT.ToTensor()(img, tgt_pt)
        arr_pt, tgt_pt2 = PT.Normalize(mean, std)(arr_pt, tgt_pt)

        pt_np = arr_pt.permute(1, 2, 0).numpy()
        np.testing.assert_allclose(arr_k, pt_np, atol=1e-5)
        np.testing.assert_allclose(
            tgt_k["boxes"], tgt_pt2["boxes"].numpy(), atol=1e-5
        )

    def test_none_target(self):
        img = _make_pil_image(50, 50)
        arr, _ = K.to_tensor()(img, _make_target_np(1, 50, 50))
        out, tgt = K.normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(arr, None)
        assert tgt is None
        # (pixel - 0.5) / 0.5 for pixel=1.0 → 1.0; pixel=0.0 → -1.0
        assert out.min() >= -1.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_box_conversion_xyxy_to_cxcywh(self):
        w, h = 200, 100
        img = _make_pil_image(w, h)
        tgt = {
            "boxes": np.array([[20, 10, 80, 50]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
            "area": np.array([2400], dtype=np.float32),
            "iscrowd": np.zeros(1, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        arr, _ = K.to_tensor()(img, _copy_np_target(tgt))
        _, tgt_out = K.normalize([0.5], [0.5])(arr, _copy_np_target(tgt))

        # Expected: cx=(20+80)/2/200=0.25, cy=(10+50)/2/100=0.3
        #           cw=(80-20)/200=0.3,    ch=(50-10)/100=0.4
        expected = np.array([[0.25, 0.3, 0.3, 0.4]], dtype=np.float32)
        np.testing.assert_allclose(tgt_out["boxes"], expected, atol=1e-5)

    def test_empty_boxes(self):
        w, h = 100, 80
        img = _make_pil_image(w, h)
        tgt = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "area": np.zeros(0, dtype=np.float32),
            "iscrowd": np.zeros(0, dtype=np.int64),
            "image_id": np.array([1], dtype=np.int64),
            "size": np.array([h, w], dtype=np.int64),
            "orig_size": np.array([h, w], dtype=np.int64),
        }
        arr, _ = K.to_tensor()(img, _copy_np_target(tgt))
        _, tgt_out = K.normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(
            arr, _copy_np_target(tgt)
        )
        assert tgt_out["boxes"].shape == (0, 4)


# =========================================================================
# Tests: Compose
# =========================================================================


class TestCompose:

    def test_empty_compose(self):
        img = _make_pil_image()
        tgt = _make_target_np()
        out_img, out_tgt = K.compose([])(img, _copy_np_target(tgt))
        assert out_img is img
        np.testing.assert_array_equal(tgt["boxes"], out_tgt["boxes"])

    def test_single_transform(self):
        img = _make_pil_image()
        tgt_np = _make_target_np()

        random.seed(SEED)
        img1, tgt1 = K.compose([K.random_horizontal_flip(p=1.0)])(
            img, _copy_np_target(tgt_np)
        )
        img2, tgt2 = K.random_horizontal_flip(p=1.0)(
            img, _copy_np_target(tgt_np)
        )
        _compare_images_pil(img1, img2)
        np.testing.assert_allclose(tgt1["boxes"], tgt2["boxes"], atol=1e-6)

    def test_chain_parity(self):
        img = _make_pil_image(200, 150)
        tgt_np = _make_target_np(3, 200, 150)
        tgt_pt = _make_target_pt(tgt_np)

        k_pipeline = K.compose([
            K.random_horizontal_flip(p=1.0),
            K.random_resize([300]),
        ])
        pt_pipeline = PT.Compose([
            PT.RandomHorizontalFlip(p=1.0),
            PT.RandomResize([300]),
        ])

        img_k, tgt_k = k_pipeline(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)


# =========================================================================
# Tests: compute_multi_scale_scales
# =========================================================================


class TestComputeMultiScaleScales:

    def test_parity(self):
        for res in [384, 512, 560, 576, 704]:
            for expanded in [False, True]:
                for ps in [12, 14, 16]:
                    for nw in [1, 2, 4]:
                        k = compute_multi_scale_scales(res, expanded, ps, nw)
                        p = pt_compute_multi_scale_scales(
                            res, expanded, ps, nw
                        )
                        assert k == p, (
                            f"Mismatch at res={res}, exp={expanded}, "
                            f"ps={ps}, nw={nw}: keras={k} vs pt={p}"
                        )

    def test_minimum_filtering(self):
        scales = compute_multi_scale_scales(
            resolution=128, expanded_scales=True, patch_size=16, num_windows=4
        )
        minimum = 16 * 4 * 2  # 128
        assert all(s >= minimum for s in scales)

    def test_sorted_output(self):
        for res in [384, 560]:
            scales = compute_multi_scale_scales(res, True, 14, 4)
            assert scales == sorted(scales)


# =========================================================================
# Tests: ConvertCoco
# =========================================================================


class TestConvertCoco:

    def test_basic(self):
        img = _make_pil_image(200, 150)
        anns = [
            {
                "bbox": [10, 20, 30, 40],
                "category_id": 1,
                "area": 1200.0,
                "iscrowd": 0,
                "id": 1,
            },
            {
                "bbox": [50, 60, 25, 35],
                "category_id": 2,
                "area": 875.0,
                "iscrowd": 0,
                "id": 2,
            },
        ]

        _, tgt_k = K_ConvertCoco()(
            img, {"image_id": 42, "annotations": anns}
        )
        _, tgt_pt = PT_ConvertCoco()(
            img, {"image_id": 42, "annotations": anns}
        )

        np.testing.assert_allclose(
            tgt_k["boxes"], tgt_pt["boxes"].numpy(), atol=1e-6
        )
        np.testing.assert_array_equal(
            tgt_k["labels"], tgt_pt["labels"].numpy()
        )
        np.testing.assert_allclose(
            tgt_k["area"], tgt_pt["area"].numpy(), atol=1e-6
        )

    def test_empty_annotations(self):
        img = _make_pil_image(200, 150)
        _, tgt_k = K_ConvertCoco()(
            img, {"image_id": 1, "annotations": []}
        )
        assert tgt_k["boxes"].shape == (0, 4)
        assert tgt_k["labels"].shape == (0,)

    def test_iscrowd_filtered(self):
        img = _make_pil_image(200, 150)
        anns = [
            {
                "bbox": [10, 20, 30, 40],
                "category_id": 1,
                "area": 1200.0,
                "iscrowd": 1,
                "id": 1,
            },
            {
                "bbox": [50, 60, 25, 35],
                "category_id": 2,
                "area": 875.0,
                "iscrowd": 0,
                "id": 2,
            },
        ]
        _, tgt_k = K_ConvertCoco()(
            img, {"image_id": 1, "annotations": anns}
        )
        _, tgt_pt = PT_ConvertCoco()(
            img, {"image_id": 1, "annotations": anns}
        )

        np.testing.assert_allclose(
            tgt_k["boxes"], tgt_pt["boxes"].numpy(), atol=1e-6
        )
        assert tgt_k["boxes"].shape[0] == 1

    def test_degenerate_box_filtered(self):
        img = _make_pil_image(200, 150)
        anns = [
            {
                "bbox": [10, 20, 0, 40],  # width=0 → degenerate
                "category_id": 1,
                "area": 0.0,
                "iscrowd": 0,
                "id": 1,
            },
            {
                "bbox": [50, 60, 25, 35],
                "category_id": 2,
                "area": 875.0,
                "iscrowd": 0,
                "id": 2,
            },
        ]
        _, tgt_k = K_ConvertCoco()(
            img, {"image_id": 1, "annotations": anns}
        )
        _, tgt_pt = PT_ConvertCoco()(
            img, {"image_id": 1, "annotations": anns}
        )

        np.testing.assert_allclose(
            tgt_k["boxes"], tgt_pt["boxes"].numpy(), atol=1e-6
        )
        assert tgt_k["boxes"].shape[0] == 1

    def test_xywh_to_xyxy_conversion(self):
        img = _make_pil_image(200, 150)
        anns = [
            {
                "bbox": [10, 20, 30, 40],
                "category_id": 1,
                "area": 1200.0,
                "iscrowd": 0,
                "id": 1,
            },
        ]
        _, tgt = K_ConvertCoco()(img, {"image_id": 1, "annotations": anns})
        expected = np.array([[10, 20, 40, 60]], dtype=np.float32)
        np.testing.assert_allclose(tgt["boxes"], expected, atol=1e-6)

    def test_all_iscrowd(self):
        img = _make_pil_image(200, 150)
        anns = [
            {"bbox": [10, 20, 30, 40], "category_id": 1,
             "area": 1200.0, "iscrowd": 1, "id": 1},
            {"bbox": [50, 60, 25, 35], "category_id": 2,
             "area": 875.0, "iscrowd": 1, "id": 2},
        ]
        _, tgt = K_ConvertCoco()(img, {"image_id": 1, "annotations": anns})
        assert tgt["boxes"].shape == (0, 4)

    def test_box_clipping_to_image(self):
        img = _make_pil_image(100, 80)
        anns = [
            {
                "bbox": [80, 60, 50, 50],  # extends beyond 100x80
                "category_id": 1,
                "area": 2500.0,
                "iscrowd": 0,
                "id": 1,
            },
        ]
        _, tgt_k = K_ConvertCoco()(img, {"image_id": 1, "annotations": anns})
        _, tgt_pt = PT_ConvertCoco()(img, {"image_id": 1, "annotations": anns})

        np.testing.assert_allclose(
            tgt_k["boxes"], tgt_pt["boxes"].numpy(), atol=1e-6
        )
        # x2 clipped to 100, y2 clipped to 80
        assert tgt_k["boxes"][0, 2] <= 100
        assert tgt_k["boxes"][0, 3] <= 80


# =========================================================================
# Tests: end-to-end pipelines
# =========================================================================


class TestFullTrainPipeline:

    def test_square_div_64_train(self):
        img = _make_pil_image(640, 480)
        tgt_np = _make_target_np(4, 640, 480)
        tgt_pt = _make_target_pt(tgt_np)

        resolution = 560
        keras_pipeline = make_coco_transforms_square_div_64(
            "train",
            resolution,
            multi_scale=False,
            expanded_scales=False,
            skip_random_resize=True,
            patch_size=14,
            num_windows=4,
        )
        pt_pipeline = pt_make_coco_transforms_square_div_64(
            "train",
            resolution,
            multi_scale=False,
            expanded_scales=False,
            skip_random_resize=True,
            patch_size=14,
            num_windows=4,
        )

        random.seed(SEED)
        img_k, tgt_k = keras_pipeline(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_tensor(img_k, img_pt, atol=1e-5)
        _compare_target(tgt_k, tgt_pt2, atol=1e-5)

    def test_square_div_64_val(self):
        img = _make_pil_image(640, 480)
        tgt_np = _make_target_np(4, 640, 480)
        tgt_pt = _make_target_pt(tgt_np)

        resolution = 560
        keras_pipeline = make_coco_transforms_square_div_64("val", resolution)
        pt_pipeline = pt_make_coco_transforms_square_div_64("val", resolution)

        random.seed(SEED)
        img_k, tgt_k = keras_pipeline(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_tensor(img_k, img_pt, atol=1e-5)
        _compare_target(tgt_k, tgt_pt2, atol=1e-5)

    def test_square_div_64_multi_scale(self):
        img = _make_pil_image(640, 480)
        tgt_np = _make_target_np(4, 640, 480)
        tgt_pt = _make_target_pt(tgt_np)

        resolution = 560
        keras_pipeline = make_coco_transforms_square_div_64(
            "train",
            resolution,
            multi_scale=True,
            expanded_scales=False,
            skip_random_resize=True,
            patch_size=14,
            num_windows=4,
        )
        pt_pipeline = pt_make_coco_transforms_square_div_64(
            "train",
            resolution,
            multi_scale=True,
            expanded_scales=False,
            skip_random_resize=True,
            patch_size=14,
            num_windows=4,
        )

        random.seed(SEED)
        img_k, tgt_k = keras_pipeline(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_tensor(img_k, img_pt, atol=1e-5)
        _compare_target(tgt_k, tgt_pt2, atol=1e-5)


class TestFullAspectRatioPipeline:

    def test_val(self):
        img = _make_pil_image(640, 480)
        tgt_np = _make_target_np(4, 640, 480)
        tgt_pt = _make_target_pt(tgt_np)

        keras_pipeline = make_coco_transforms("val", 560)
        pt_pipeline = pt_make_coco_transforms("val", 560)

        random.seed(SEED)
        img_k, tgt_k = keras_pipeline(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_tensor(img_k, img_pt, atol=1e-5)
        _compare_target(tgt_k, tgt_pt2, atol=1e-5)

    def test_val_speed(self):
        img = _make_pil_image(640, 480)
        tgt_np = _make_target_np(4, 640, 480)
        tgt_pt = _make_target_pt(tgt_np)

        keras_pipeline = make_coco_transforms("val_speed", 560)
        pt_pipeline = pt_make_coco_transforms("val_speed", 560)

        random.seed(SEED)
        img_k, tgt_k = keras_pipeline(img, _copy_np_target(tgt_np))

        random.seed(SEED)
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_tensor(img_k, img_pt, atol=1e-5)
        _compare_target(tgt_k, tgt_pt2, atol=1e-5)

    @pytest.mark.parametrize("resolution", [384, 512, 560, 640])
    def test_val_various_resolutions(self, resolution):
        img = _make_pil_image(640, 480)
        tgt_np = _make_target_np(3, 640, 480)
        tgt_pt = _make_target_pt(tgt_np)

        keras_pipeline = make_coco_transforms("val", resolution)
        pt_pipeline = pt_make_coco_transforms("val", resolution)

        img_k, tgt_k = keras_pipeline(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = pt_pipeline(img, tgt_pt)

        _compare_images_tensor(img_k, img_pt, atol=1e-5)
        _compare_target(tgt_k, tgt_pt2, atol=1e-5)


# =========================================================================
# Tests: multi-image stress / consistency
# =========================================================================


class TestMultipleImages:

    @pytest.mark.parametrize(
        "w,h",
        [(50, 50), (1, 1), (640, 480), (480, 640), (100, 1), (1, 100)],
    )
    def test_hflip_various_shapes(self, w, h):
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(2, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.hflip(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.hflip(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    @pytest.mark.parametrize(
        "w,h", [(100, 80), (200, 200), (300, 100), (50, 300)]
    )
    def test_resize_various_shapes(self, w, h):
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(2, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.resize(
            img, _copy_np_target(tgt_np), 150, max_size=500
        )
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, 150, max_size=500)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    @pytest.mark.parametrize(
        "w,h", [(200, 150), (150, 200), (300, 300)]
    )
    def test_square_resize_various_shapes(self, w, h):
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(2, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.square_resize([256])(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.SquareResize([256])(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)


# =========================================================================
# Tests: many-box scenarios
# =========================================================================


class TestManyBoxes:

    def test_hflip_many_boxes(self):
        w, h, n = 300, 200, 50
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(n, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.hflip(img, _copy_np_target(tgt_np))
        img_pt, tgt_pt2 = PT.hflip(img, tgt_pt)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_crop_many_boxes(self):
        w, h, n = 300, 200, 50
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(n, w, h)
        tgt_pt = _make_target_pt(tgt_np)
        region = (20, 30, 150, 200)

        img_k, tgt_k = K.crop(img, _copy_np_target(tgt_np), region)
        img_pt, tgt_pt2 = PT.crop(img, tgt_pt, region)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)

    def test_resize_many_boxes(self):
        w, h, n = 300, 200, 50
        img = _make_pil_image(w, h)
        tgt_np = _make_target_np(n, w, h)
        tgt_pt = _make_target_pt(tgt_np)

        img_k, tgt_k = K.resize(img, _copy_np_target(tgt_np), 100)
        img_pt, tgt_pt2 = PT.resize(img, tgt_pt, 100)

        _compare_images_pil(img_k, img_pt)
        _compare_target(tgt_k, tgt_pt2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
