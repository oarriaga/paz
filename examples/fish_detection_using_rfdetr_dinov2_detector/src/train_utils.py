import os
import sys
import json
import math
import logging
import argparse
import time
from pathlib import Path

import numpy as np

# ImageNet channel statistics — DINOv2 pretraining distribution
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype="float32")

# ---------------------------------------------------------------------------
# Ensure the paz package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAZ_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PAZ_ROOT not in sys.path:
    sys.path.insert(0, _PAZ_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Variant class mapping (direct imports — no VARIANT_REGISTRY)
# ---------------------------------------------------------------------------
from paz.models.detection.dino_v2_object_detection.detr import (
    RFDETRBase,
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge,
    RFDETRXLarge,
    RFDETR2XLarge,
)

VARIANT_MAP = {
    "RFDETRBase": RFDETRBase,
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge": RFDETRLarge,
    "RFDETRXLarge": RFDETRXLarge,
    "RFDETR2XLarge": RFDETR2XLarge,
}

VARIANT_FRIENDLY_NAME = {
    "RFDETRBase": "rfdetr_base",
    "RFDETRNano": "rfdetr_nano",
    "RFDETRSmall": "rfdetr_small",
    "RFDETRMedium": "rfdetr_medium",
    "RFDETRLarge": "rfdetr_large",
    "RFDETRXLarge": "rfdetr_xlarge",
    "RFDETR2XLarge": "rfdetr_2xlarge",
}

_DEFAULT_EXPERIMENTS_ROOT = os.path.join(_SCRIPT_DIR, "experiments")


# ---------------------------------------------------------------------------
# Dataset -> COCO-format conversion
# ---------------------------------------------------------------------------


def prepare_coco_dataset(ds, output_dir, val_split=0.1, seed=42):
    """Convert a ``DeepFishDataset`` to the COCO directory structure
    expected by the RF-DETR high-level API.

    Creates lightweight **symlinks** for images (no data copying).

    Parameters
    ----------
    ds : DeepFishDataset
        Dataset object exposing ``_img_ids``, ``_annotations``,
        ``_class_to_id``, ``class_names``, and ``get_image_path(img_id)``.
        Annotations must use normalised coordinate keys
        (``x_min_norm``, ``x_max_norm``, ``y_min_norm``, ``y_max_norm``).
    output_dir : str
    val_split : float
    seed : int

    Returns
    -------
    coco_dir : str
    train_indices : list[int]
    val_indices : list[int]
    """
    from PIL import Image as PILImage

    coco_dir = os.path.join(output_dir, "_coco_format")

    rng = np.random.RandomState(seed)
    all_indices = rng.permutation(len(ds)).tolist()
    n_val = max(1, int(len(all_indices) * val_split))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(ds.class_names)
    ]

    for split_name, indices in [("train", train_indices),
                                ("val", val_indices)]:
        split_dir = os.path.join(coco_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        images_list = []
        annotations_list = []
        ann_id = 0

        for idx in indices:
            img_id = ds._img_ids[idx]

            # Locate image on disk
            img_path = ds.get_image_path(img_id)
            if img_path is None:
                continue

            # Read dimensions without decoding pixels
            with PILImage.open(img_path) as pil_img:
                orig_w, orig_h = pil_img.size

            filename = os.path.basename(img_path)
            images_list.append({
                "id": idx,
                "file_name": filename,
                "width": orig_w,
                "height": orig_h,
            })

            # Create symlink into the split directory
            dst = os.path.join(split_dir, filename)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(img_path), dst)

            # Convert normalised annotations -> COCO bbox [x, y, w, h] (absolute)
            rows = ds._annotations.get(img_id, [])
            for row in rows:
                label_str = row["label_l1"].strip()
                if label_str not in ds._class_to_id:
                    continue

                x_min = float(row["x_min_norm"]) * orig_w
                x_max = float(row["x_max_norm"]) * orig_w
                y_min = float(row["y_min_norm"]) * orig_h
                y_max = float(row["y_max_norm"]) * orig_h

                x_min = max(0.0, min(x_min, orig_w))
                x_max = max(0.0, min(x_max, orig_w))
                y_min = max(0.0, min(y_min, orig_h))
                y_max = max(0.0, min(y_max, orig_h))
                if x_max <= x_min or y_max <= y_min:
                    continue

                bw, bh = x_max - x_min, y_max - y_min
                annotations_list.append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": ds._class_to_id[label_str],
                    "bbox": [float(x_min), float(y_min), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1

        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump({
                "images": images_list,
                "annotations": annotations_list,
                "categories": categories,
            }, f)

    return coco_dir, train_indices, val_indices


# ---------------------------------------------------------------------------
# Object-detection validation metrics (pure NumPy — no framework dep)
# ---------------------------------------------------------------------------


def _softmax(x):
    """Numerically stable softmax over the last axis.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray
        Probability distribution along ``axis=-1``.
    """
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _box_cxcywh_to_xyxy(boxes):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4)
        Normalised centre-format boxes.

    Returns
    -------
    np.ndarray, shape (N, 4)
        Corner-format boxes.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _compute_iou_matrix(boxes_a, boxes_b):
    """Compute the pairwise IoU matrix between two box sets.

    Parameters
    ----------
    boxes_a : np.ndarray, shape (Na, 4)
        Corner-format (x1, y1, x2, y2) boxes.
    boxes_b : np.ndarray, shape (Nb, 4)
        Corner-format (x1, y1, x2, y2) boxes.

    Returns
    -------
    np.ndarray, shape (Na, Nb)
        IoU value for each (a, b) pair.
    """
    # (Na, 1, 4) vs (1, Nb, 4)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(
        inter_y2 - inter_y1, 0)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def _compute_ap(recalls, precisions):
    """Compute Average Precision using 11-point interpolation.

    Parameters
    ----------
    recalls : np.ndarray
        Cumulative recall curve.
    precisions : np.ndarray
        Cumulative precision curve.

    Returns
    -------
    float
        Interpolated AP value (PASCAL VOC style).
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p_at_r = precisions[recalls >= t]
        ap += p_at_r.max() if len(p_at_r) > 0 else 0.0
    return ap / 11.0


def validate_epoch_full(
    model,
    criterion,
    dataset,
    indices,
    batch_size,
    num_classes,
    class_names,
    conf_threshold=0.3,
    iou_threshold=0.5,
    max_batches=None,
    logger=None,
    prefix="val",
):
    """Run full object-detection evaluation with proper metrics.

    Parameters
    ----------
    prefix : str
        Key prefix for the returned metrics dict (e.g. ``"val"`` or
        ``"train"``).  All returned scalar keys will be
        ``{prefix}_loss``, ``{prefix}_mAP_50``, etc.

    Returns
    -------
    metrics : dict
        Keys: {prefix}_loss, {prefix}_loss_ce, {prefix}_loss_bbox,
              {prefix}_loss_giou, {prefix}_mAP_50, {prefix}_mAP_50_95,
              {prefix}_precision, {prefix}_recall, {prefix}_f1,
              {prefix}_accuracy, {prefix}_num_gt_boxes,
              {prefix}_num_pred_boxes, per_class_precision,
              per_class_recall, per_class_f1, per_class_ap50.
    """
    from keras import ops
    from training_helpers import make_batches

    val_losses = []
    val_losses_ce = []
    val_losses_bbox = []
    val_losses_giou = []

    # Per-class accumulators for AP computation
    # For each class: list of (confidence, is_tp) tuples
    per_class_detections = {c: [] for c in range(num_classes)}
    per_class_num_gt = np.zeros(num_classes, dtype=np.int64)

    total_gt_boxes = 0
    total_pred_boxes = 0

    for images, targets in make_batches(dataset, indices, batch_size,
                                        max_batches):
        # Apply ImageNet normalisation — DeepFishDataset yields raw [0, 1]
        # float32 images; the model expects inputs pre-normalised to the
        # DINOv2 pretraining distribution.
        images = (images - _IMAGENET_MEAN) / _IMAGENET_STD
        outputs = model(images, training=False)

        # ---- Loss -------------------------------------------------------
        saved_group_detr = getattr(criterion, "group_detr", 1)
        criterion.group_detr = 1
        try:
            loss_dict = criterion(outputs, targets)
        finally:
            criterion.group_detr = saved_group_detr
        weight_dict = criterion.weight_dict
        total_loss = sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict if k in weight_dict
        )
        val_losses.append(float(ops.convert_to_numpy(total_loss)))

        # ---- Per-component losses (for diagnostic logging) ---------------
        try:
            val_losses_ce.append(float(ops.convert_to_numpy(
                loss_dict.get("loss_ce", 0.0))))
            val_losses_bbox.append(float(ops.convert_to_numpy(
                loss_dict.get("loss_bbox", 0.0))))
            val_losses_giou.append(float(ops.convert_to_numpy(
                loss_dict.get("loss_giou", 0.0))))
        except Exception:
            pass

        # ---- Detection metrics ------------------------------------------
        pred_logits = ops.convert_to_numpy(outputs["pred_logits"])
        pred_boxes_cxcywh = ops.convert_to_numpy(outputs["pred_boxes"])
        B = pred_logits.shape[0]

        for b in range(B):
            # Ground truth
            gt_labels = np.asarray(targets[b]["labels"]).flatten()
            gt_boxes = np.asarray(targets[b]["boxes"])
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes.reshape(-1, 4)
            gt_boxes_xyxy = _box_cxcywh_to_xyxy(gt_boxes) if len(gt_boxes) > 0 \
                else np.zeros((0, 4))
            num_gt = len(gt_labels)
            total_gt_boxes += num_gt

            for lbl in gt_labels:
                if 0 <= lbl < num_classes:
                    per_class_num_gt[lbl] += 1

            # Predictions — RF-DETR uses sigmoid (not softmax) for classification
            probs = 1.0 / (1.0 + np.exp(-pred_logits[b][:, :num_classes]))
            max_probs = probs.max(axis=-1)
            pred_cls = probs.argmax(axis=-1)

            # Filter by confidence
            keep = max_probs > conf_threshold
            kept_scores = max_probs[keep]
            kept_cls = pred_cls[keep]
            kept_boxes_cxcywh = pred_boxes_cxcywh[b][keep]
            kept_boxes_xyxy = _box_cxcywh_to_xyxy(kept_boxes_cxcywh) \
                if len(kept_boxes_cxcywh) > 0 else np.zeros((0, 4))
            total_pred_boxes += len(kept_scores)

            # Sort by confidence (descending)
            sort_idx = np.argsort(-kept_scores)
            kept_scores = kept_scores[sort_idx]
            kept_cls = kept_cls[sort_idx]
            kept_boxes_xyxy = kept_boxes_xyxy[sort_idx]

            # Match predictions to ground truth (greedy, per-class)
            gt_matched = np.zeros(num_gt, dtype=bool)

            for d in range(len(kept_scores)):
                d_cls = int(kept_cls[d])
                d_score = float(kept_scores[d])
                if d_cls < 0 or d_cls >= num_classes:
                    continue

                is_tp = False
                best_iou = 0.0
                if num_gt > 0:
                    ious = _compute_iou_matrix(
                        kept_boxes_xyxy[d:d+1], gt_boxes_xyxy)[0]
                    # Find best matching GT of same class
                    best_iou = 0.0
                    best_gt_idx = -1
                    for g in range(num_gt):
                        if gt_matched[g]:
                            continue
                        if int(gt_labels[g]) != d_cls:
                            continue
                        if ious[g] > best_iou:
                            best_iou = ious[g]
                            best_gt_idx = g
                    if best_gt_idx >= 0 and best_iou >= iou_threshold:
                        is_tp = True
                        gt_matched[best_gt_idx] = True

                per_class_detections[d_cls].append(
                    (d_score, is_tp, best_iou))

    # ---- Aggregate metrics -----------------------------------------------
    avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

    per_class_precision = np.zeros(num_classes)
    per_class_recall = np.zeros(num_classes)
    per_class_f1 = np.zeros(num_classes)
    per_class_ap50 = np.zeros(num_classes)

    active_classes = []
    for c in range(num_classes):
        n_gt = per_class_num_gt[c]
        dets = per_class_detections[c]
        if n_gt == 0 and len(dets) == 0:
            continue
        active_classes.append(c)

        # Sort detections by confidence (descending)
        dets.sort(key=lambda x: -x[0])
        scores_arr = np.array([d[0] for d in dets])
        tp_arr = np.array([d[1] for d in dets], dtype=bool)

        if n_gt == 0:
            # All detections are FP
            per_class_precision[c] = 0.0
            per_class_recall[c] = 0.0
            per_class_f1[c] = 0.0
            per_class_ap50[c] = 0.0
            continue

        # Cumulative TP / FP
        cum_tp = np.cumsum(tp_arr.astype(np.float64))
        cum_fp = np.cumsum((~tp_arr).astype(np.float64))

        rec = cum_tp / n_gt
        prec = cum_tp / (cum_tp + cum_fp)

        # AP@50
        per_class_ap50[c] = _compute_ap(rec, prec)

        # Precision/Recall/F1 at the confidence threshold
        total_tp = cum_tp[-1] if len(cum_tp) > 0 else 0
        total_fp = cum_fp[-1] if len(cum_fp) > 0 else 0
        p = total_tp / max(total_tp + total_fp, 1)
        r = total_tp / max(n_gt, 1)
        per_class_precision[c] = p
        per_class_recall[c] = r
        per_class_f1[c] = 2 * p * r / max(p + r, 1e-8)

    # Macro-average over active classes
    if active_classes:
        macro_prec = per_class_precision[active_classes].mean()
        macro_rec = per_class_recall[active_classes].mean()
        macro_f1 = per_class_f1[active_classes].mean()
        mAP_50 = per_class_ap50[active_classes].mean()
    else:
        macro_prec = macro_rec = macro_f1 = mAP_50 = 0.0

    # mAP@50:95 — proper computation at 10 IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_per_iou = []
    for iou_t in iou_thresholds:
        aps = []
        for c in active_classes:
            n_gt = per_class_num_gt[c]
            if n_gt == 0:
                aps.append(0.0)
                continue
            dets = per_class_detections[c]
            # Re-evaluate TP/FP at this IoU threshold using stored best_iou
            tp_at_iou = np.array([d[2] >= iou_t for d in dets], dtype=bool)
            cum_tp = np.cumsum(tp_at_iou.astype(np.float64))
            cum_fp = np.cumsum((~tp_at_iou).astype(np.float64))
            rec = cum_tp / n_gt
            prec = cum_tp / (cum_tp + cum_fp)
            aps.append(_compute_ap(rec, prec))
        ap_per_iou.append(np.mean(aps) if aps else 0.0)
    mAP_50_95 = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0

    # Overall accuracy = total TP / total GT
    total_tp_all = sum(
        sum(1 for _, tp, _ in per_class_detections[c] if tp)
        for c in range(num_classes)
    )
    accuracy = total_tp_all / max(total_gt_boxes, 1)

    metrics = {
        f"{prefix}_loss": avg_val_loss,
        f"{prefix}_loss_ce": float(np.mean(val_losses_ce)) if val_losses_ce else 0.0,
        f"{prefix}_loss_bbox": float(np.mean(val_losses_bbox)) if val_losses_bbox else 0.0,
        f"{prefix}_loss_giou": float(np.mean(val_losses_giou)) if val_losses_giou else 0.0,
        f"{prefix}_mAP_50": float(mAP_50),
        f"{prefix}_mAP_50_95": float(mAP_50_95),
        f"{prefix}_precision": float(macro_prec),
        f"{prefix}_recall": float(macro_rec),
        f"{prefix}_f1": float(macro_f1),
        f"{prefix}_accuracy": float(accuracy),
        f"{prefix}_num_gt_boxes": int(total_gt_boxes),
        f"{prefix}_num_pred_boxes": int(total_pred_boxes),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_ap50": per_class_ap50,
    }

    # Log per-class breakdown
    if logger is not None:
        logger.info("  --- Per-class metrics (AP@50 / Prec / Rec / F1) ---")
        for c in active_classes:
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            logger.info(
                "    %-25s AP50=%.3f  P=%.3f  R=%.3f  F1=%.3f  (GT=%d)",
                name,
                per_class_ap50[c],
                per_class_precision[c],
                per_class_recall[c],
                per_class_f1[c],
                per_class_num_gt[c],
            )

    return metrics


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(output_dir):
    """Configure logging to both stdout and a file.

    Creates ``output_dir/output.txt`` in append mode and attaches
    both a stream handler (stdout) and a file handler to the root
    logger.

    Parameters
    ----------
    output_dir : str
        Experiment directory; created if it does not exist.

    Returns
    -------
    logging.Logger
        Logger named ``'train_v2'``.
    """
    os.makedirs(output_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(
        os.path.join(output_dir, "output.txt"), mode="a",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return logging.getLogger("train_v2")
