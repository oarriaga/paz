"""Dataset utilities for the Keras 3 RF-DETR replication.

Public API mirrors the PyTorch ``rfdetr.datasets`` package.
"""

from paz.models.detection.dino_v2_object_detection.datasets.coco import (
    build as build_coco,
    build_roboflow,
    CocoDetection,
    COCOBatchLoader,
    compute_multi_scale_scales,
)


def build_dataset(image_set, args, resolution):
    """Build a :class:`CocoDetection` dataset.

    Args:
        image_set (str): ``'train'``, ``'val'``, or ``'test'``.
        args: Namespace-like object with dataset configuration attributes.
        resolution (int): Target image resolution.

    Returns:
        CocoDetection: The constructed dataset.
    """
    dataset_file = getattr(args, "dataset_file", "roboflow")
    if dataset_file == "coco":
        return build_coco(image_set, args, resolution)
    if dataset_file in ("roboflow", "coco_json"):
        return build_roboflow(image_set, args, resolution)
    raise ValueError(f"dataset {dataset_file} not supported")


def get_coco_api_from_dataset(dataset):
    """Return the underlying ``pycocotools.coco.COCO`` object, or ``None``."""
    if hasattr(dataset, "coco"):
        return dataset.coco
    return None
