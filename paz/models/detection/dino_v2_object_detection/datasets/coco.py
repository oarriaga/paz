import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import paz.models.detection.dino_v2_object_detection.datasets.transforms as T

# Host-side by design: this whole module is numpy/PIL/pycocotools data
# loading that runs on CPU before batching.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_RESIZE = 1333
CROP_SCALES = [400, 500, 600]
EXPANDED_OFFSETS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
DEFAULT_OFFSETS = [-3, -2, -1, 0, 1, 2, 3, 4]
TARGET_KEYS = ("boxes", "labels", "image_id", "area", "iscrowd", "orig_size", "size")  # fmt: skip


def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):  # fmt: skip
    base = resolution // (patch_size * num_windows)
    offsets = EXPANDED_OFFSETS if expanded_scales else DEFAULT_OFFSETS
    window = patch_size * num_windows
    proposed = [(base + offset) * window for offset in offsets]
    return [size for size in proposed if size >= window * 2]


def read_annotation_boxes(annotations, width, height):
    boxes = [obj["bbox"] for obj in annotations]
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    # xywh -> xyxy, clipped to the image extent
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
    return boxes


def build_coco_masks(annotations, keep, height, width):
    empty = np.zeros((0, height, width), dtype=bool)
    masks = empty
    try:
        if len(annotations) > 0 and "segmentation" in annotations[0]:
            polygons = [obj.get("segmentation", []) for obj in annotations]
            decoded = convert_poly_to_mask(polygons, height, width)
            if decoded.size > 0:
                masks = decoded[keep].astype(bool)
    except ImportError:
        masks = empty
    return masks


def build_coco_target(image_id, annotations, keep, boxes, height, width):
    labels = [obj["category_id"] for obj in annotations]
    area = np.array([obj["area"] for obj in annotations], dtype=np.float32)
    crowd = [obj.get("iscrowd", 0) for obj in annotations]
    size = np.array([int(height), int(width)], dtype=np.int64)
    values = (boxes[keep], np.array(labels, dtype=np.int64)[keep], np.array([image_id], dtype=np.int64), area[keep], np.array(crowd, dtype=np.int64)[keep], size, size)  # fmt: skip
    return dict(zip(TARGET_KEYS, values))


def convert_coco(include_masks=False):
    def apply(image, target):
        width, height = image.size
        annotations = [obj for obj in target["annotations"] if not obj.get("iscrowd", 0)]  # fmt: skip
        boxes = read_annotation_boxes(annotations, width, height)
        wide = boxes[:, 2] > boxes[:, 0]
        keep = (boxes[:, 3] > boxes[:, 1]) & wide
        args = (target["image_id"], annotations, keep, boxes)
        converted = build_coco_target(*args, height, width)
        if include_masks:
            args = (annotations, keep, height, width)
            converted["masks"] = build_coco_masks(*args)
        return image, converted

    return apply


def decode_polygon(coco_mask_util, polygons, height, width):
    try:
        rles = coco_mask_util.frPyObjects(polygons, height, width)
    except Exception:
        rles = polygons
    mask = coco_mask_util.decode(rles)
    if mask.ndim < 3:
        mask = mask[..., np.newaxis]
    return mask.any(axis=2).astype(np.uint8)


def convert_poly_to_mask(segmentations, height, width):
    import pycocotools.mask as coco_mask_util
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            masks.append(np.zeros((height, width), dtype=np.uint8))
        else:
            args = (coco_mask_util, polygons, height, width)
            masks.append(decode_polygon(*args))
    if len(masks) == 0:
        result = np.zeros((0, height, width), dtype=np.uint8)
    else:
        result = np.stack(masks, axis=0)
    return result


# Kept as a class: it owns the pycocotools index plus the id list and is
# consumed through the len/getitem protocol by the batch loaders.
class CocoDetection:
    def __init__(self, img_folder, ann_file, transforms=None, include_masks=False):  # fmt: skip
        from pycocotools.coco import COCO

        self.img_folder = str(img_folder)
        self.coco = COCO(str(ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = transforms
        self.prepare = convert_coco(include_masks=include_masks)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.img_folder, info["file_name"])
        image = Image.open(path).convert("RGB")
        annotations = self.coco.loadAnns(annotation_ids)
        target = {"image_id": image_id, "annotations": annotations}
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


def build_normalizer():
    return T.compose([T.to_tensor(), T.normalize(IMAGENET_MEAN, IMAGENET_STD)])


def resolve_transform_scales(resolution, multi_scale, expanded_scales, skip_random_resize, patch_size, num_windows):  # fmt: skip
    scales = [resolution]
    if multi_scale:
        args = (resolution, expanded_scales, patch_size, num_windows)
        scales = compute_multi_scale_scales(*args)
    if multi_scale and skip_random_resize:
        scales = [scales[-1]]
    return scales


def build_train_crop_branch(resize_transform):
    steps = [T.random_resize(CROP_SCALES), T.random_size_crop(384, 600)]
    return T.compose(steps + [resize_transform])


def build_train_pipeline(resize_transform, normalizer):
    branch = build_train_crop_branch(resize_transform)
    selector = T.random_select(resize_transform, branch)
    return T.compose([T.random_horizontal_flip(), selector, normalizer])


def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):  # fmt: skip
    normalizer = build_normalizer()
    args = (resolution, multi_scale, expanded_scales, skip_random_resize)
    scales = resolve_transform_scales(*args, patch_size, num_windows)
    if image_set == "train":
        resize_transform = T.random_resize(scales, max_size=MAX_RESIZE)
        pipeline = build_train_pipeline(resize_transform, normalizer)
    elif image_set == "val":
        resize_transform = T.random_resize([resolution], max_size=MAX_RESIZE)
        pipeline = T.compose([resize_transform, normalizer])
    elif image_set == "val_speed":
        pipeline = T.compose([T.square_resize([resolution]), normalizer])
    else:
        raise ValueError(f"unknown {image_set}")
    return pipeline


def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):  # fmt: skip
    normalizer = build_normalizer()
    args = (resolution, multi_scale, expanded_scales, skip_random_resize)
    scales = resolve_transform_scales(*args, patch_size, num_windows)
    if image_set == "train":
        pipeline = build_train_pipeline(T.square_resize(scales), normalizer)
    elif image_set in ("val", "test", "val_speed"):
        pipeline = T.compose([T.square_resize([resolution]), normalizer])
    else:
        raise ValueError(f"unknown {image_set}")
    return pipeline


def select_transform_factory(args, default_square):
    square = getattr(args, "square_resize_div_64", default_square)
    return make_coco_transforms_square_div_64 if square else make_coco_transforms  # fmt: skip


def build_transform_pipeline(factory, image_set, resolution, args):
    keys = ("multi_scale", "expanded_scales", "skip_random_resize", "patch_size", "num_windows")  # fmt: skip
    padded = getattr(args, "do_random_resize_via_padding", False)
    values = (getattr(args, "multi_scale", False), getattr(args, "expanded_scales", False), not padded, getattr(args, "patch_size", 16), getattr(args, "num_windows", 4))  # fmt: skip
    return factory(image_set, resolution, **dict(zip(keys, values)))


def build_coco_paths(root, mode):
    keys = ("train", "val", "test")
    annotations = root / "annotations"
    values = ((root / "train2017", annotations / f"{mode}_train2017.json"), (root / "val2017", annotations / f"{mode}_val2017.json"), (root / "test2017", annotations / "image_info_test-dev2017.json"))  # fmt: skip
    return dict(zip(keys, values))


def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    paths = build_coco_paths(root, "instances")
    img_folder, ann_file = paths[image_set.split("_")[0]]
    factory = select_transform_factory(args, False)
    transforms = build_transform_pipeline(factory, image_set, resolution, args)
    return CocoDetection(img_folder, ann_file, transforms=transforms)


def build_roboflow_paths(root):
    keys = ("train", "val", "test")
    folders = (root / "train", root / "valid", root / "test")
    values = [(f, f / "_annotations.coco.json") for f in folders]
    return dict(zip(keys, values))


def build_roboflow(image_set, args, resolution):
    root = Path(getattr(args, "dataset_dir", "."))
    assert root.exists(), f"provided Roboflow path {root} does not exist"
    paths = build_roboflow_paths(root)
    img_folder, ann_file = paths[image_set.split("_")[0]]
    factory = select_transform_factory(args, True)
    transforms = build_transform_pipeline(factory, image_set, resolution, args)
    include_masks = getattr(args, "segmentation_head", False)
    args = (img_folder, ann_file)
    return CocoDetection(*args, transforms=transforms, include_masks=include_masks)  # fmt: skip


def collate_with_padding(images):
    shapes = [image.shape[:2] for image in images]
    max_height = max(height for height, _ in shapes)
    max_width = max(width for _, width in shapes)
    same = all(s == (max_height, max_width) for s in shapes)
    if same:
        batched, mask = np.stack(images, axis=0).astype(np.float32), None
    else:
        shape = (len(images), max_height, max_width)
        batched = np.zeros(shape + (3,), dtype=np.float32)
        mask = np.ones(shape, dtype=bool)  # True = padded
        for index, image in enumerate(images):
            height, width = image.shape[:2]
            batched[index, :height, :width, :] = image
            mask[index, :height, :width] = False
    return batched, mask


def build_sample_indices(dataset, replacement, num_samples, shuffle):
    total = len(dataset)
    if replacement and num_samples is not None:
        indices = np.random.choice(total, size=num_samples, replace=True)
    elif shuffle:
        indices = np.random.permutation(total)
    else:
        indices = np.arange(total)
    return indices


def collate_batch(images, targets):
    batched, mask = collate_with_padding(images)
    return ((batched, mask), targets) if mask is not None else (batched, targets)  # fmt: skip


# Kept as classes: both own iteration state (sampling order, worker pool)
# and are consumed through the len/iter protocol.
class COCOBatchLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False, replacement=False, num_samples=None):  # fmt: skip
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.replacement = replacement
        self.num_samples = num_samples

    def __len__(self):
        total = len(self.dataset)
        if self.replacement and self.num_samples is not None:
            total = self.num_samples
        if self.drop_last:
            length = total // self.batch_size
        else:
            length = math.ceil(total / self.batch_size)
        return length

    def __iter__(self):
        args = (self.dataset, self.replacement, self.num_samples, self.shuffle)
        indices = build_sample_indices(*args)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            samples = [self.dataset[int(index)] for index in batch]
            images = [image for image, _ in samples]
            yield collate_batch(images, [target for _, target in samples])


class PrefetchBatchLoader:
    def __init__(self, base_loader, num_workers=2):
        self.base_loader = base_loader
        self.num_workers = max(1, num_workers)

    @property
    def dataset(self):
        return self.base_loader.dataset

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        loader = self.base_loader
        args = (loader.dataset, loader.replacement, loader.num_samples)
        indices = build_sample_indices(*args, loader.shuffle)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start in range(0, len(indices), loader.batch_size):
                batch = indices[start : start + loader.batch_size]
                if loader.drop_last and len(batch) < loader.batch_size:
                    break
                yield load_prefetch_batch(executor, loader.dataset, batch)


def load_prefetch_batch(executor, dataset, batch):
    def load_sample(index):
        return dataset[int(index)]

    futures = [executor.submit(load_sample, index) for index in batch]
    samples = [future.result() for future in futures]
    images = [image for image, _ in samples]
    return collate_batch(images, [target for _, target in samples])
