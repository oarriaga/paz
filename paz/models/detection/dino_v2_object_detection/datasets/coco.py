import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import paz.models.detection.dino_v2_object_detection.datasets.transforms as T


# ---------------------------------------------------------------------------
# Multi-scale helpers
# ---------------------------------------------------------------------------


def compute_multi_scale_scales(
    resolution, expanded_scales=False, patch_size=16, num_windows=4
):
    """Compute the set of multi-scale training resolutions.

    Args:
        resolution (int): Base resolution.
        expanded_scales (bool): Use wider offset range.
        patch_size (int): ViT patch size.
        num_windows (int): Number of attention windows.

    Returns:
        list[int]: Sorted candidate scales.
    """
    base = resolution // (patch_size * num_windows)
    offsets = (
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        if expanded_scales
        else [-3, -2, -1, 0, 1, 2, 3, 4]
    )
    scales = [base + o for o in offsets]
    proposed = [s * patch_size * num_windows for s in scales]
    return [s for s in proposed if s >= patch_size * num_windows * 2]


# ---------------------------------------------------------------------------
# COCO annotation conversion
# ---------------------------------------------------------------------------


class ConvertCoco:
    """Convert raw COCO annotation dicts into the standard target dict.

    """

    def __init__(self, include_masks=False):
        self.include_masks = include_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        anno = target["annotations"]

        # Filter iscrowd
        anno = [
            obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0
        ]

        boxes = [obj["bbox"] for obj in anno]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        # xywh → xyxy
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)

        classes = np.array(
            [obj["category_id"] for obj in anno], dtype=np.int64
        )

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        area = np.array([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.array(
            [obj.get("iscrowd", 0) for obj in anno], dtype=np.int64
        )
        area = area[keep]
        iscrowd = iscrowd[keep]

        out = {
            "boxes": boxes,
            "labels": classes,
            "image_id": np.array([image_id], dtype=np.int64),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": np.array([int(h), int(w)], dtype=np.int64),
            "size": np.array([int(h), int(w)], dtype=np.int64),
        }

        if self.include_masks:
            try:
                import pycocotools.mask as coco_mask_util

                if len(anno) > 0 and "segmentation" in anno[0]:
                    segs = [obj.get("segmentation", []) for obj in anno]
                    masks = _convert_poly_to_mask(segs, h, w)
                    if masks.size > 0:
                        out["masks"] = masks[keep].astype(bool)
                    else:
                        out["masks"] = np.zeros((0, h, w), dtype=bool)
                else:
                    out["masks"] = np.zeros((0, h, w), dtype=bool)
            except ImportError:
                out["masks"] = np.zeros((0, h, w), dtype=bool)

        return image, out


def _convert_poly_to_mask(segmentations, h, w):
    """Polygon segmentations → binary masks ``(N, H, W)`` uint8."""
    import pycocotools.mask as coco_mask_util

    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue
        try:
            rles = coco_mask_util.frPyObjects(polygons, h, w)
        except Exception:
            rles = polygons
        mask = coco_mask_util.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        mask = mask.any(axis=2).astype(np.uint8)
        masks.append(mask)
    if len(masks) == 0:
        return np.zeros((0, h, w), dtype=np.uint8)
    return np.stack(masks, axis=0)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class CocoDetection:
    """Minimal COCO detection dataset (no torchvision dependency).

    Loads images from *img_folder* and annotations from *ann_file* using
    ``pycocotools``.

    Each ``__getitem__`` call returns ``(image, target)`` after applying
    optional transforms.  The image is a PIL Image (before transforms) or
    whatever the transform pipeline outputs.
    """

    def __init__(self, img_folder, ann_file, transforms=None, include_masks=False):
        from pycocotools.coco import COCO

        self.img_folder = str(img_folder)
        self.coco = COCO(str(ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = transforms
        self.prepare = ConvertCoco(include_masks=include_masks)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(
            os.path.join(self.img_folder, img_info["file_name"])
        ).convert("RGB")
        target = {"image_id": img_id, "annotations": anns}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def make_coco_transforms(
    image_set,
    resolution,
    multi_scale=False,
    expanded_scales=False,
    skip_random_resize=False,
    patch_size=16,
    num_windows=4,
):
    """Aspect-ratio-preserving transform pipeline."""
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    scales = [resolution]
    if multi_scale:
        scales = compute_multi_scale_scales(
            resolution, expanded_scales, patch_size, num_windows
        )
        if skip_random_resize:
            scales = [scales[-1]]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [T.RandomResize([resolution], max_size=1333), normalize]
        )
    if image_set == "val_speed":
        return T.Compose([T.SquareResize([resolution]), normalize])

    raise ValueError(f"unknown {image_set}")


def make_coco_transforms_square_div_64(
    image_set,
    resolution,
    multi_scale=False,
    expanded_scales=False,
    skip_random_resize=False,
    patch_size=16,
    num_windows=4,
):
    """Square-resize transform pipeline (default for Roboflow datasets)."""
    normalize = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    scales = [resolution]
    if multi_scale:
        scales = compute_multi_scale_scales(
            resolution, expanded_scales, patch_size, num_windows
        )
        if skip_random_resize:
            scales = [scales[-1]]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.SquareResize(scales),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.SquareResize(scales),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set in ("val", "test", "val_speed"):
        return T.Compose([T.SquareResize([resolution]), normalize])

    raise ValueError(f"unknown {image_set}")


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build(image_set, args, resolution):
    """Build a standard COCO dataset."""
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (
            root / "train2017",
            root / "annotations" / f"{mode}_train2017.json",
        ),
        "val": (
            root / "val2017",
            root / "annotations" / f"{mode}_val2017.json",
        ),
        "test": (
            root / "test2017",
            root / "annotations" / "image_info_test-dev2017.json",
        ),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)

    factory = (
        make_coco_transforms_square_div_64
        if square_resize_div_64
        else make_coco_transforms
    )
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=factory(
            image_set,
            resolution,
            multi_scale=getattr(args, "multi_scale", False),
            expanded_scales=getattr(args, "expanded_scales", False),
            skip_random_resize=not getattr(
                args, "do_random_resize_via_padding", False
            ),
            patch_size=getattr(args, "patch_size", 16),
            num_windows=getattr(args, "num_windows", 4),
        ),
    )
    return dataset


def build_roboflow(image_set, args, resolution):
    """Build a Roboflow-exported COCO-format dataset."""
    root = Path(getattr(args, "dataset_dir", "."))
    assert root.exists(), f"provided Roboflow path {root} does not exist"
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root / "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }

    img_folder, ann_file = PATHS[image_set.split("_")[0]]

    square_resize_div_64 = getattr(args, "square_resize_div_64", True)
    include_masks = getattr(args, "segmentation_head", False)

    factory = (
        make_coco_transforms_square_div_64
        if square_resize_div_64
        else make_coco_transforms
    )
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=factory(
            image_set,
            resolution,
            multi_scale=getattr(args, "multi_scale", False),
            expanded_scales=getattr(args, "expanded_scales", False),
            skip_random_resize=not getattr(
                args, "do_random_resize_via_padding", False
            ),
            patch_size=getattr(args, "patch_size", 16),
            num_windows=getattr(args, "num_windows", 4),
        ),
        include_masks=include_masks,
    )
    return dataset


# ---------------------------------------------------------------------------
# Padded collation (NestedTensor-like)
# ---------------------------------------------------------------------------


def _collate_with_padding(images):
    """Pad a list of images to the largest spatial size and build a mask.

    When all images already have the same shape, the mask is ``None``
    (no padding was needed).  Otherwise a boolean mask of shape
    ``(B, max_H, max_W)`` is returned where ``True`` marks *padded*
    (invalid) pixels.

    Args:
        images (list[np.ndarray]): Per-image arrays of shape ``(H, W, 3)``.

    Returns:
        tuple: ``(batched_images, mask)`` — ``batched_images`` is
            ``(B, max_H, max_W, 3)`` float32; ``mask`` is
            ``(B, max_H, max_W)`` bool or ``None``.
    """
    shapes = [img.shape[:2] for img in images]
    max_h = max(h for h, w in shapes)
    max_w = max(w for h, w in shapes)

    all_same = all(h == max_h and w == max_w for h, w in shapes)
    if all_same:
        return np.stack(images, axis=0).astype(np.float32), None

    B = len(images)
    batched = np.zeros((B, max_h, max_w, 3), dtype=np.float32)
    mask = np.ones((B, max_h, max_w), dtype=bool)  # True = padded
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        batched[i, :h, :w, :] = img
        mask[i, :h, :w] = False
    return batched, mask


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------


class COCOBatchLoader:
    """Batch iterator over a :class:`CocoDetection` dataset.

    Yields ``(images_np, targets)`` — ``images_np`` is a
    ``(B, H, W, 3)`` float32 array and ``targets`` is a list of dicts.

    The images within a batch must all be the same spatial size.  Because
    the transform pipeline already resizes them to a fixed square
    resolution (via ``SquareResize``), this is guaranteed.

    Args:
        dataset: The dataset to iterate over.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle indices each epoch.
        drop_last (bool): Drop the last incomplete batch.
        replacement (bool): Sample with replacement.
        num_samples (int or None): Total samples when using replacement.
    """

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False,
                 replacement=False, num_samples=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.replacement = replacement
        self.num_samples = num_samples

    def __len__(self):
        if self.replacement and self.num_samples is not None:
            n = self.num_samples
        else:
            n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if self.replacement and self.num_samples is not None:
            indices = np.random.choice(
                n, size=self.num_samples, replace=True
            )
        elif self.shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break
            images, targets = [], []
            for idx in batch_idx:
                img, tgt = self.dataset[int(idx)]
                images.append(img)
                targets.append(tgt)
            images_np, mask = _collate_with_padding(images)
            if mask is not None:
                yield (images_np, mask), targets
            else:
                yield images_np, targets


class PrefetchBatchLoader:
    """Prefetching wrapper that loads batches in background threads.

    Wraps a ``COCOBatchLoader`` and uses a thread pool to prefetch samples
    ahead of time, keeping the GPU fed during training.

    Args:
        base_loader (COCOBatchLoader): The underlying batch loader.
        num_workers (int): Number of prefetch threads.
    """

    def __init__(self, base_loader, num_workers=2):
        self.base_loader = base_loader
        self.num_workers = max(1, num_workers)

    @property
    def dataset(self):
        return self.base_loader.dataset

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        dataset = self.base_loader.dataset
        batch_size = self.base_loader.batch_size
        n = len(dataset)

        if self.base_loader.replacement and self.base_loader.num_samples:
            indices = np.random.choice(
                n, size=self.base_loader.num_samples, replace=True
            )
        elif self.base_loader.shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)

        def _load_sample(idx):
            return dataset[int(idx)]

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                if (self.base_loader.drop_last
                        and len(batch_idx) < batch_size):
                    break
                futures = [
                    executor.submit(_load_sample, idx) for idx in batch_idx
                ]
                images, targets = [], []
                for f in futures:
                    img, tgt = f.result()
                    images.append(img)
                    targets.append(tgt)
                images_np, mask = _collate_with_padding(images)
                if mask is not None:
                    yield (images_np, mask), targets
                else:
                    yield images_np, targets
