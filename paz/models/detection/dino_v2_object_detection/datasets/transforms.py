"""
Transforms and data augmentation for both image + bbox.

Every transform is a callable ``(PIL.Image, target_dict) -> (image, target_dict)``
where ``target_dict`` values are **numpy** arrays (float32 / int64 / bool).
"""
import random
from collections.abc import Sequence
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image

# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------


def crop(image, target, region):
    """Crop *image* and adjust *target* bounding-boxes / masks.

    Args:
        image (PIL.Image.Image): Input image.
        target (dict): Target dict with ``boxes`` in xyxy pixel coords.
        region (tuple): ``(top, left, height, width)``.

    Returns:
        tuple: ``(cropped_image, updated_target)``.
    """
    i, j, h, w = region
    cropped_image = image.crop((j, i, j + w, i + h))

    target = target.copy()
    target["size"] = np.array([h, w], dtype=np.int64)

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"].copy()
        max_size = np.array([w, h], dtype=np.float32)
        cropped_boxes = boxes - np.array([j, i, j, i], dtype=np.float32)
        cropped_boxes = cropped_boxes.reshape(-1, 2, 2)
        cropped_boxes = np.minimum(cropped_boxes, max_size)
        cropped_boxes = np.clip(cropped_boxes, 0, None)
        area = np.prod(cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :], axis=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # Remove elements whose boxes / masks have zero area
    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        else:
            keep = target["masks"].reshape(target["masks"].shape[0], -1).any(axis=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    """Horizontally flip *image* and mirror *target* boxes.

    Args:
        image (PIL.Image.Image): Input image.
        target (dict): Target dict with ``boxes`` in xyxy pixel coords.

    Returns:
        tuple: ``(flipped_image, updated_target)``.
    """
    flipped_image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"].copy()
        boxes = (
            boxes[:, [2, 1, 0, 3]]
            * np.array([-1, 1, -1, 1], dtype=np.float32)
            + np.array([w, 0, w, 0], dtype=np.float32)
        )
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"][:, :, ::-1].copy()

    return flipped_image, target


def _get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)


def _get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
        return size[::-1]
    else:
        return _get_size_with_aspect_ratio(image_size, size, max_size)


def resize(image, target, size, max_size=None):
    """Resize *image* and scale *target* boxes / masks accordingly.

    Args:
        image (PIL.Image.Image): Input image.
        target (dict or None): Target dict, may be ``None``.
        size (int or tuple): Target short-edge or ``(w, h)`` tuple.
        max_size (int or None): Upper-bound on the long edge.

    Returns:
        tuple: ``(resized_image, updated_target)``.
    """
    new_size = _get_size(image.size, size, max_size)  # (h, w)
    rescaled_image = image.resize((new_size[1], new_size[0]), PIL.Image.BILINEAR)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig)
        for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * np.array(
            [ratio_width, ratio_height, ratio_width, ratio_height],
            dtype=np.float32,
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = new_size
    target["size"] = np.array([h, w], dtype=np.int64)

    if "masks" in target:
        masks = target["masks"]  # (N, H_old, W_old) bool/uint8
        if masks.shape[0] > 0:
            resized = []
            for m in masks:
                m_pil = PIL.Image.fromarray(m.astype(np.uint8) * 255)
                m_pil = m_pil.resize((w, h), PIL.Image.NEAREST)
                resized.append(np.asarray(m_pil) > 127)
            target["masks"] = np.stack(resized, axis=0)
        else:
            target["masks"] = np.zeros((0, h, w), dtype=bool)

    return rescaled_image, target


def pad(image, target, padding):
    """Pad *image* on the bottom-right and update *target* size.

    Args:
        image (PIL.Image.Image): Input image.
        target (dict or None): Target dict.
        padding (tuple): ``(pad_right, pad_bottom)``.

    Returns:
        tuple: ``(padded_image, updated_target)``.
    """
    pad_right, pad_bottom = padding
    w, h = image.size
    new_w, new_h = w + pad_right, h + pad_bottom
    padded_image = PIL.Image.new(image.mode, (new_w, new_h), color=0)
    padded_image.paste(image, (0, 0))

    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = np.array([new_h, new_w], dtype=np.int64)
    if "masks" in target:
        masks = target["masks"]
        padded_masks = np.zeros(
            (masks.shape[0], new_h, new_w), dtype=masks.dtype
        )
        padded_masks[:, :h, :w] = masks
        target["masks"] = padded_masks
    return padded_image, target


# ---------------------------------------------------------------------------
# Transform classes
# ---------------------------------------------------------------------------


class RandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, img, target):
        w, h = img.size
        th, tw = self.size
        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger than image ({h}, {w})"
            )
        if h == th and w == tw:
            return crop(img, target, (0, 0, h, w))
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        return crop(img, target, (top, left, th, tw))


class RandomSizeCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        img_w, img_h = img.size
        w = random.randint(self.min_size, min(img_w, self.max_size))
        h = random.randint(self.min_size, min(img_h, self.max_size))
        # Same as T.RandomCrop.get_params(img, [h, w])
        top = random.randint(0, img_h - h)
        left = random.randint(0, img_w - w)
        return crop(img, target, (top, left, h, w))


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class SquareResize:
    """Resize to a square of randomly chosen size (ignores aspect ratio)."""

    def __init__(self, sizes):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        rescaled_img = img.resize((size, size), PIL.Image.BILINEAR)
        w, h = rescaled_img.size  # both == size

        if target is None:
            return rescaled_img, None

        ratios = tuple(
            float(s) / float(s_orig)
            for s, s_orig in zip(rescaled_img.size, img.size)
        )
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * np.array(
                [ratio_width, ratio_height, ratio_width, ratio_height],
                dtype=np.float32,
            )
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        target["size"] = np.array([h, w], dtype=np.int64)

        if "masks" in target:
            masks = target["masks"]
            if masks.shape[0] > 0:
                resized = []
                for m in masks:
                    m_pil = PIL.Image.fromarray(m.astype(np.uint8) * 255)
                    m_pil = m_pil.resize((w, h), PIL.Image.NEAREST)
                    resized.append(np.asarray(m_pil) > 127)
                target["masks"] = np.stack(resized, axis=0)
            else:
                target["masks"] = np.zeros((0, h, w), dtype=bool)

        return rescaled_img, target


class RandomPad:
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect:
    """Randomly select between two transform pipelines."""

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    """Convert PIL Image to ``(H, W, C)`` float32 numpy array in [0, 1]."""

    def __call__(self, img, target):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr, target


class Normalize:
    """Apply ImageNet normalisation and convert boxes to normalised cxcywh.

    Works on ``(H, W, C)`` float32 numpy arrays (Keras convention).
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, target=None):
        image = (image - self.mean) / self.std
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[:2]  # (H, W, C)
        if "boxes" in target:
            boxes = target["boxes"]
            # xyxy -> cxcywh
            x0, y0, x1, y1 = (
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2],
                boxes[:, 3],
            )
            b = np.stack(
                [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)],
                axis=-1,
            )
            b = b / np.array([w, h, w, h], dtype=np.float32)
            target["boxes"] = b
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t}")
        lines.append(")")
        return "\n".join(lines)
