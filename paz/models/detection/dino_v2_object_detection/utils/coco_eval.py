import os
import contextlib
import copy
import functools
from types import SimpleNamespace

import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

EVALUATOR_METHODS = ("update", "accumulate", "summarize", "prepare", "prepare_for_coco_detection", "prepare_for_coco_segmentation")  # fmt: skip
SUMMARY_FORMAT = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"  # fmt: skip
# (average_precision, iou_threshold, area_range, max_detections_slot)
SUMMARY_ROWS = ((1, None, "all", 2), (1, 0.5, "all", 2), (1, 0.75, "all", 2), (1, None, "small", 2), (1, None, "medium", 2), (1, None, "large", 2), (0, None, "all", 0), (0, None, "all", 1), (0, None, "all", 2), (0, None, "small", 2), (0, None, "medium", 2), (0, None, "large", 2))  # fmt: skip
# Mask logits are thresholded at 0.5 for COCO, unlike 0.0 in post-processing.
MASK_THRESHOLD = 0.5


def CocoEvaluator(coco_gt, iou_types, max_dets=100):
    assert isinstance(iou_types, (list, tuple))
    ns = SimpleNamespace()
    ns.coco_gt = copy.deepcopy(coco_gt)
    ns.max_dets = max_dets
    ns.iou_types = iou_types
    ns.coco_eval = build_coco_evaluations(ns.coco_gt, iou_types, max_dets)
    ns.img_ids = []
    ns.eval_imgs = {iou_type: [] for iou_type in iou_types}
    functions = (update_coco_evaluator, accumulate_coco_results, summarize_coco_results, prepare_coco_results, prepare_coco_detections, prepare_coco_segmentations)  # fmt: skip
    for name, function in zip(EVALUATOR_METHODS, functions):
        setattr(ns, name, functools.partial(function, ns))
    return ns


def build_coco_evaluations(coco_gt, iou_types, max_dets):
    evaluations = {}
    for iou_type in iou_types:
        evaluation = COCOeval(coco_gt, iouType=iou_type)
        evaluation.params.maxDets = [1, 10, max_dets]
        evaluations[iou_type] = evaluation
    return evaluations


def load_detection_results(coco_gt, results):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            loaded = COCO.loadRes(coco_gt, results) if results else COCO()
    return loaded


def update_coco_evaluator(ns, predictions):
    img_ids = list(np.unique(list(predictions.keys())))
    ns.img_ids.extend(img_ids)
    for iou_type in ns.iou_types:
        results = ns.prepare(predictions, iou_type)
        coco_eval = ns.coco_eval[iou_type]
        coco_eval.cocoDt = load_detection_results(ns.coco_gt, results)
        coco_eval.params.imgIds = list(img_ids)
        ns.eval_imgs[iou_type].append(evaluate_coco_images(coco_eval)[1])


def accumulate_coco_results(ns):
    for iou_type in ns.iou_types:
        merged = np.concatenate(ns.eval_imgs[iou_type], 2)
        ns.eval_imgs[iou_type] = merged
        create_common_coco_eval(ns.coco_eval[iou_type], ns.img_ids, merged)
    for coco_eval in ns.coco_eval.values():
        coco_eval.accumulate()


def summarize_coco_results(ns):
    for iou_type, coco_eval in ns.coco_eval.items():
        print("IoU metric: {}".format(iou_type))
        patched_summarize(coco_eval)


def prepare_coco_results(ns, predictions, iou_type):
    if iou_type == "bbox":
        results = ns.prepare_for_coco_detection(predictions)
    elif iou_type == "segm":
        results = ns.prepare_for_coco_segmentation(predictions)
    else:
        raise ValueError("Unknown iou type {}".format(iou_type))
    return results


def to_list(value):
    return value.tolist() if hasattr(value, "tolist") else value


# ns is bound by functools.partial so every prepare_* helper shares the
# evaluator's dispatch signature, even when it reads nothing from it.
def prepare_coco_detections(ns, predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = to_list(convert_to_xywh(prediction["boxes"]))
        scores = to_list(prediction["scores"])
        labels = to_list(prediction["labels"])
        for index, box in enumerate(boxes):
            entry = {"image_id": original_id, "category_id": labels[index]}
            entry["bbox"] = box
            entry["score"] = scores[index]
            coco_results.append(entry)
    return coco_results


def encode_masks(masks):
    rles = []
    for mask in masks:
        packed = np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
        rle = mask_util.encode(packed)[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles


def prepare_coco_segmentations(ns, predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        scores = to_list(prediction["scores"])
        labels = to_list(prediction["labels"])
        rles = encode_masks(prediction["masks"] > MASK_THRESHOLD)
        for index, rle in enumerate(rles):
            entry = {"image_id": original_id, "category_id": labels[index]}
            entry["segmentation"] = rle
            entry["score"] = scores[index]
            coco_results.append(entry)
    return coco_results


def convert_to_xywh(boxes):
    boxes = np.array(boxes)
    xmin, ymin = boxes[:, 0], boxes[:, 1]
    xmax, ymax = boxes[:, 2], boxes[:, 3]
    return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=1)


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, index = np.unique(np.array(img_ids), return_index=True)
    coco_eval.evalImgs = list(eval_imgs[..., index].flatten())
    coco_eval.params.imgIds = list(img_ids)
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def normalize_eval_params(coco_eval):
    params = coco_eval.params
    if params.useSegm is not None:
        params.iouType = "segm" if params.useSegm == 1 else "bbox"
    params.imgIds = list(np.unique(params.imgIds))
    if params.useCats:
        params.catIds = list(np.unique(params.catIds))
    params.maxDets = sorted(params.maxDets)
    coco_eval.params = params
    return params


def select_iou_function(coco_eval, iou_type):
    if iou_type == "keypoints":
        compute = coco_eval.computeOks
    else:
        compute = coco_eval.computeIoU
    return compute


def evaluate_coco_images(coco_eval):
    params = normalize_eval_params(coco_eval)
    coco_eval._prepare()
    category_ids = params.catIds if params.useCats else [-1]
    compute_iou = select_iou_function(coco_eval, params.iouType)
    coco_eval.ious = {(image_id, category_id): compute_iou(image_id, category_id) for image_id in params.imgIds for category_id in category_ids}  # fmt: skip
    max_detections = params.maxDets[-1]
    evaluated = [coco_eval.evaluateImg(image_id, category_id, area_range, max_detections) for category_id in category_ids for area_range in params.areaRng for image_id in params.imgIds]  # fmt: skip
    shape = (len(category_ids), len(params.areaRng), len(params.imgIds))
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    return params.imgIds, np.asarray(evaluated).reshape(shape)


def patched_summarize(coco_eval):
    if not coco_eval.eval:
        raise Exception("Please run accumulate() first")
    coco_eval.stats = summarize_detections(coco_eval)


def format_iou_range(params, iou_threshold):
    if iou_threshold is None:
        formatted = "{:0.2f}:{:0.2f}".format(params.iouThrs[0], params.iouThrs[-1])  # fmt: skip
    else:
        formatted = "{:0.2f}".format(iou_threshold)
    return formatted


def select_summary_scores(coco_eval, average_precision, iou_threshold, area_index, max_index):  # fmt: skip
    params = coco_eval.params
    key = "precision" if average_precision else "recall"
    scores = coco_eval.eval[key]
    if iou_threshold is not None:
        scores = scores[np.where(iou_threshold == params.iouThrs)[0]]
    if average_precision:
        scores = scores[:, :, :, area_index, max_index]
    else:
        scores = scores[:, :, area_index, max_index]
    return scores


def coco_summarize(coco_eval, ap=1, iouThr=None, areaRng="all", maxDets=100):
    params = coco_eval.params
    area_index = [i for i, label in enumerate(params.areaRngLbl) if label == areaRng]  # fmt: skip
    max_index = [i for i, value in enumerate(params.maxDets) if value == maxDets]  # fmt: skip
    args = (coco_eval, ap == 1, iouThr, area_index, max_index)
    scores = select_summary_scores(*args)
    mean_score = np.mean(scores[scores > -1]) if len(scores[scores > -1]) else -1  # fmt: skip
    title = "Average Precision" if ap == 1 else "Average Recall"
    kind = "(AP)" if ap == 1 else "(AR)"
    iou_range = format_iou_range(params, iouThr)
    print(SUMMARY_FORMAT.format(title, kind, iou_range, areaRng, maxDets, mean_score))  # fmt: skip
    return mean_score


def summarize_detections(coco_eval):
    stats = np.zeros((len(SUMMARY_ROWS),))
    for index, row in enumerate(SUMMARY_ROWS):
        average_precision, iou_threshold, area_range, max_slot = row
        max_detections = coco_eval.params.maxDets[max_slot]
        args = (coco_eval, average_precision, iou_threshold, area_range)
        stats[index] = coco_summarize(*args, max_detections)
    return stats
