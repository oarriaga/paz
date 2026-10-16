from paz.models.detection.dino_v2_object_detection.datasets.coco import (
    build as build_coco,
    build_roboflow,
    CocoDetection,
    COCOBatchLoader,
    compute_multi_scale_scales,
)


__all__ = [
    "build_dataset",
    "get_coco_api_from_dataset",
    "build_coco",
    "build_roboflow",
    "CocoDetection",
    "COCOBatchLoader",
    "compute_multi_scale_scales",
]


def build_dataset(image_set, args, resolution):
    dataset_file = getattr(args, "dataset_file", "roboflow")
    if dataset_file == "coco":
        dataset = build_coco(image_set, args, resolution)
    elif dataset_file in ("roboflow", "coco_json"):
        dataset = build_roboflow(image_set, args, resolution)
    else:
        raise ValueError(f"dataset {dataset_file} not supported")
    return dataset


def get_coco_api_from_dataset(dataset):
    coco = None
    if hasattr(dataset, "coco"):
        coco = dataset.coco
    return coco
