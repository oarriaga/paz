import random

import numpy as np
import PIL.Image

# This whole module is host-side by design: PIL/numpy augmentation runs on
# CPU before batching, so nothing here should be converted to keras.ops.

CROP_FIELDS = ["labels", "area", "iscrowd"]
MASK_THRESHOLD = 127


def resize_mask(mask, width, height):
    image = PIL.Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((width, height), PIL.Image.NEAREST)
    return np.asarray(resized) > MASK_THRESHOLD


def resize_masks(masks, width, height):
    if masks.shape[0] == 0:
        resized = np.zeros((0, height, width), dtype=bool)
    else:
        stack = [resize_mask(mask, width, height) for mask in masks]
        resized = np.stack(stack, axis=0)
    return resized


def scale_target(target, image, rescaled, size):
    pairs = zip(rescaled.size, image.size)
    ratio_width, ratio_height = [float(a) / float(b) for a, b in pairs]
    target = target.copy()
    if "boxes" in target:
        scale = [ratio_width, ratio_height, ratio_width, ratio_height]
        target["boxes"] = target["boxes"] * np.array(scale, dtype=np.float32)
    if "area" in target:
        target["area"] = target["area"] * (ratio_width * ratio_height)
    height, width = size
    target["size"] = np.array([height, width], dtype=np.int64)
    if "masks" in target:
        target["masks"] = resize_masks(target["masks"], width, height)
    return target


def crop_target_boxes(target, top, left, height, width):
    boxes = target["boxes"].copy()
    max_size = np.array([width, height], dtype=np.float32)
    offset = np.array([left, top, left, top], dtype=np.float32)
    cropped = (boxes - offset).reshape(-1, 2, 2)
    cropped = np.clip(np.minimum(cropped, max_size), 0, None)
    target["area"] = np.prod(cropped[:, 1, :] - cropped[:, 0, :], axis=1)
    target["boxes"] = cropped.reshape(-1, 4)
    return target


def build_crop_keep_mask(target):
    keep = None
    if "boxes" in target:
        corners = target["boxes"].reshape(-1, 2, 2)
        keep = np.all(corners[:, 1, :] > corners[:, 0, :], axis=1)
    elif "masks" in target:
        masks = target["masks"]
        keep = masks.reshape(masks.shape[0], -1).any(axis=1)
    return keep


def crop(image, target, region):
    top, left, height, width = region
    cropped_image = image.crop((left, top, left + width, top + height))
    target = target.copy()
    target["size"] = np.array([height, width], dtype=np.int64)
    fields = list(CROP_FIELDS)
    if "boxes" in target:
        target = crop_target_boxes(target, top, left, height, width)
        fields.append("boxes")
    if "masks" in target:
        rows, columns = slice(top, top + height), slice(left, left + width)
        target["masks"] = target["masks"][:, rows, columns]
        fields.append("masks")
    keep = build_crop_keep_mask(target)
    for field in fields if keep is not None else []:
        target[field] = target[field][keep]
    return cropped_image, target


def hflip(image, target):
    flipped_image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    width = image.size[0]
    target = target.copy()
    if "boxes" in target:
        mirrored = target["boxes"].copy()[:, [2, 1, 0, 3]]
        scale = np.array([-1, 1, -1, 1], dtype=np.float32)
        offset = np.array([width, 0, width, 0], dtype=np.float32)
        target["boxes"] = mirrored * scale + offset
    if "masks" in target:
        target["masks"] = target["masks"][:, :, ::-1].copy()
    return flipped_image, target


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    width, height = image_size
    if max_size is not None:
        smallest = float(min(image_size))
        largest = float(max(image_size))
        if largest / smallest * size > max_size:
            size = int(round(max_size * smallest / largest))
    short_side_matches = (width <= height and width == size)
    short_side_matches = short_side_matches or (height <= width and height == size)  # fmt: skip
    result = (height, width)
    if not short_side_matches and width < height:
        result = (int(size * height / width), size)
    elif not short_side_matches:
        result = (size, int(size * width / height))
    return result


def get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
        resolved = size[::-1]
    else:
        resolved = get_size_with_aspect_ratio(image_size, size, max_size)
    return resolved


def resize(image, target, size, max_size=None):
    new_size = get_size(image.size, size, max_size)  # (height, width)
    rescaled_image = image.resize((new_size[1], new_size[0]), PIL.Image.BILINEAR)  # fmt: skip
    if target is not None:
        target = scale_target(target, image, rescaled_image, new_size)
    return rescaled_image, target


def pad(image, target, padding):
    pad_right, pad_bottom = padding
    width, height = image.size
    new_width, new_height = width + pad_right, height + pad_bottom
    padded_image = PIL.Image.new(image.mode, (new_width, new_height), color=0)
    padded_image.paste(image, (0, 0))
    if target is not None:
        target = target.copy()
        target["size"] = np.array([new_height, new_width], dtype=np.int64)
        if "masks" in target:
            args = (target["masks"], new_height, new_width)
            target["masks"] = pad_masks(*args, height, width)
    return padded_image, target


def pad_masks(masks, new_height, new_width, height, width):
    padded = np.zeros((masks.shape[0], new_height, new_width), dtype=masks.dtype)  # fmt: skip
    padded[:, :height, :width] = masks
    return padded


def random_crop(size):
    size = (size, size) if isinstance(size, int) else tuple(size)

    def apply(image, target):
        width, height = image.size
        target_height, target_width = size
        if height + 1 < target_height or width + 1 < target_width:
            message = f"Required crop size {size} is larger than image ({height}, {width})"  # fmt: skip
            raise ValueError(message)
        region = build_random_crop_region(height, width, size)
        return crop(image, target, region)

    return apply


def build_random_crop_region(height, width, size):
    target_height, target_width = size
    region = (0, 0, height, width)
    if height != target_height or width != target_width:
        top = random.randint(0, height - target_height)
        left = random.randint(0, width - target_width)
        region = (top, left, target_height, target_width)
    return region


def random_size_crop(min_size, max_size):
    def apply(image, target):
        image_width, image_height = image.size
        width = random.randint(min_size, min(image_width, max_size))
        height = random.randint(min_size, min(image_height, max_size))
        top = random.randint(0, image_height - height)
        left = random.randint(0, image_width - width)
        return crop(image, target, (top, left, height, width))

    return apply


def center_crop(size):
    def apply(image, target):
        image_width, image_height = image.size
        crop_height, crop_width = size
        top = int(round((image_height - crop_height) / 2.0))
        left = int(round((image_width - crop_width) / 2.0))
        return crop(image, target, (top, left, crop_height, crop_width))

    return apply


def random_horizontal_flip(p=0.5):
    def apply(image, target):
        output = image, target
        if random.random() < p:
            output = hflip(image, target)
        return output

    return apply


def random_resize(sizes, max_size=None):
    assert isinstance(sizes, (list, tuple))

    def apply(image, target=None):
        return resize(image, target, random.choice(sizes), max_size)

    return apply


def square_resize(sizes):
    assert isinstance(sizes, (list, tuple))

    def apply(image, target=None):
        size = random.choice(sizes)
        rescaled_image = image.resize((size, size), PIL.Image.BILINEAR)
        if target is not None:
            args = (target, image, rescaled_image, (size, size))
            target = scale_target(*args)
        return rescaled_image, target

    return apply


def random_pad(max_pad):
    def apply(image, target):
        padding = (random.randint(0, max_pad), random.randint(0, max_pad))
        return pad(image, target, padding)

    return apply


def random_select(transforms1, transforms2, p=0.5):
    def apply(image, target):
        if random.random() < p:
            output = transforms1(image, target)
        else:
            output = transforms2(image, target)
        return output

    return apply


def to_tensor():
    def apply(image, target):
        return np.asarray(image, dtype=np.float32) / 255.0, target

    return apply


def normalize_target_boxes(target, size):
    target = target.copy()
    if "boxes" in target:
        height, width = size
        boxes = target["boxes"]
        x0, y0 = boxes[:, 0], boxes[:, 1]
        x1, y1 = boxes[:, 2], boxes[:, 3]
        # xyxy -> cxcywh, then normalised by the image size
        centers = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        extent = np.array([width, height, width, height], dtype=np.float32)
        target["boxes"] = np.stack(centers, axis=-1) / extent
    return target


def normalize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    def apply(image, target=None):
        image = (image - mean) / std
        if target is not None:
            target = normalize_target_boxes(target, image.shape[:2])
        return image, target

    return apply


def compose(transforms):
    def apply(image, target):
        for transform in transforms:
            image, target = transform(image, target)
        return image, target

    return apply
