import os
import contextlib
import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util


class CocoEvaluator:
    """Accumulates per-image predictions and computes COCO metrics.

    Supports both ``"bbox"`` (bounding box) and ``"segm"`` (instance
    segmentation) evaluation types.

    Args:
        coco_gt (COCO): Ground-truth COCO object.
        iou_types (list[str]): Evaluation types (e.g. ``["bbox"]``).
        max_dets (int): Maximum detections per image.
    """

    def __init__(self, coco_gt: COCO, iou_types: List[str],
                 max_dets: int = 100) -> None:
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.max_dets = max_dets

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].params.maxDets = [1, 10, max_dets]

        self.img_ids: List[int] = []
        self.eval_imgs: Dict[str, List] = {k: [] for k in iou_types}

    def update(self, predictions: Dict[int, Any]) -> None:
        """Add a batch of predictions.

        Args:
            predictions: ``{image_id: {"scores": ndarray, "labels": ndarray,
                "boxes": ndarray}}`` where boxes are xyxy format.
        """
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = (
                        COCO.loadRes(self.coco_gt, results)
                        if results
                        else COCO()
                    )
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids_eval, eval_imgs = _evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def accumulate(self) -> None:
        """Accumulate per-image evaluations."""
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(
                self.eval_imgs[iou_type], 2
            )
            _create_common_coco_eval(
                self.coco_eval[iou_type],
                self.img_ids,
                self.eval_imgs[iou_type],
            )

        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self) -> None:
        """Print and store summary metrics."""
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            _patched_summarize(coco_eval)

    def prepare(self, predictions: Dict[int, Any],
                iou_type: str) -> List[Dict[str, Any]]:
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(
        self, predictions: Dict[int, Any]
    ) -> List[Dict[str, Any]]:
        """Convert detection predictions to COCO result format."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = _convert_to_xywh(boxes)
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            scores = prediction["scores"]
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            labels = prediction["labels"]
            if hasattr(labels, "tolist"):
                labels = labels.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(
        self, predictions: Dict[int, Any]
    ) -> List[Dict[str, Any]]:
        """Convert segmentation predictions to COCO result format."""
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            # Threshold at 0.5 for COCO evaluation (different from 0.0 in
            # PostProcess)
            masks = masks > 0.5

            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if hasattr(labels, "tolist"):
                labels = labels.tolist()

            rles = [
                mask_util.encode(
                    np.array(
                        mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"
                    )
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results


def _convert_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert xyxy boxes to xywh format."""
    boxes = np.array(boxes)
    xmin, ymin, xmax, ymax = (
        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    )
    return np.stack(
        (xmin, ymin, xmax - xmin, ymax - ymin), axis=1
    )


def _create_common_coco_eval(
    coco_eval: COCOeval, img_ids: List[int], eval_imgs: Any
) -> None:
    img_ids = np.array(img_ids)
    img_ids, idx = np.unique(img_ids, return_index=True)
    eval_imgs = eval_imgs[..., idx]

    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def _evaluate(coco_eval: COCOeval) -> Tuple[List[int], np.ndarray]:
    """Run per-image evaluation and return image IDs and eval results."""
    p = coco_eval.params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    coco_eval.params = p

    coco_eval._prepare()
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = coco_eval.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = coco_eval.computeOks

    coco_eval.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = coco_eval.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    evalImgs = np.asarray(evalImgs).reshape(
        len(catIds), len(p.areaRng), len(p.imgIds)
    )
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    return p.imgIds, evalImgs


def _patched_summarize(coco_eval):
    """Compute and display summary metrics with configurable max_dets."""

    def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
        p = coco_eval.params
        iStr = (
            " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ]"
            " = {:0.3f}"
        )
        titleStr = "Average Precision" if ap == 1 else "Average Recall"
        typeStr = "(AP)" if ap == 1 else "(AR)"
        iouStr = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
            if iouThr is None
            else "{:0.2f}".format(iouThr)
        )

        aind = [
            i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
        ]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            s = coco_eval.eval["precision"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = coco_eval.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(
            iStr.format(
                titleStr, typeStr, iouStr, areaRng, maxDets, mean_s
            )
        )
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1, maxDets=coco_eval.params.maxDets[2])
        stats[1] = _summarize(
            1, iouThr=0.5, maxDets=coco_eval.params.maxDets[2]
        )
        stats[2] = _summarize(
            1, iouThr=0.75, maxDets=coco_eval.params.maxDets[2]
        )
        stats[3] = _summarize(
            1, areaRng="small", maxDets=coco_eval.params.maxDets[2]
        )
        stats[4] = _summarize(
            1, areaRng="medium", maxDets=coco_eval.params.maxDets[2]
        )
        stats[5] = _summarize(
            1, areaRng="large", maxDets=coco_eval.params.maxDets[2]
        )
        stats[6] = _summarize(0, maxDets=coco_eval.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=coco_eval.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=coco_eval.params.maxDets[2])
        stats[9] = _summarize(
            0, areaRng="small", maxDets=coco_eval.params.maxDets[2]
        )
        stats[10] = _summarize(
            0, areaRng="medium", maxDets=coco_eval.params.maxDets[2]
        )
        stats[11] = _summarize(
            0, areaRng="large", maxDets=coco_eval.params.maxDets[2]
        )
        return stats

    if not coco_eval.eval:
        raise Exception("Please run accumulate() first")
    iouType = coco_eval.params.iouType
    if iouType == "segm" or iouType == "bbox":
        summarize = _summarizeDets
    coco_eval.stats = summarize()
