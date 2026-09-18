#!/usr/bin/env python
"""Experiment 8: RF-DETR Nano fine-tuning — fast fixed-shape benchmark.

Derived from Experiment 7, but intentionally configured for lower and more
stable step time while keeping the same high-level API and monitoring style.

Speed-focused changes relative to Experiment 7:
  - grad_accum_steps=1 (no micro-batch accumulation inside a logged step)
  - multi_scale=False (no batch-level random resize in the engine)
  - expanded_scales=False (paired with disabled multi-scale)
  - fixed 384x384 training shape (Nano default resolution)
  - num_workers=2 (thread-based prefetch in the native COCO loader)

This experiment exists specifically as a clean benchmark variant so the
running Experiment 7 configuration remains unchanged.
"""

import os
import sys
import json
import math
import logging
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (must come before framework imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAZ_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "src"))
if _PAZ_ROOT not in sys.path:
    sys.path.insert(0, _PAZ_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ---------------------------------------------------------------------------
# Imports — framework + project
# ---------------------------------------------------------------------------
import keras
from keras import ops

from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano
from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.main import (
    build_criterion_from_config,
)

from dataset import DeepFishDataset
from train_utils import (
    prepare_coco_dataset,
    setup_logging,
    validate_epoch_full,
)
from metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tee stdout → log file so MetricLogger per-step prints appear in output.txt
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Duplicate writes to both the original stream and a file."""

    def __init__(self, original, log_path):
        self._original = original
        self._log_file = open(log_path, "a")

    def write(self, text):
        self._original.write(text)
        self._log_file.write(text)
        self._log_file.flush()

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    def __getattr__(self, name):
        return getattr(self._original, name)



# =====================================================================
# Experiment tracker — validation, checkpoints, plots, summaries
# =====================================================================

class ExperimentTracker:
    """Full experiment tracking via callbacks, matching Experiment 7."""

    def __init__(
        self,
        model_ref,
        keras_model,
        criterion,
        dataset,
        train_indices,
        val_indices,
        num_classes,
        class_names,
        exp_dir,
        batch_size,
        total_epochs,
        conf_threshold=0.3,
        iou_threshold=0.5,
    ):
        self.model_ref = model_ref
        self.keras_model = keras_model
        self.val_model = keras_model
        self.criterion = criterion
        self.dataset = dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.num_classes = num_classes
        self.class_names = class_names
        self.exp_dir = exp_dir
        self.ckpt_dir = os.path.join(exp_dir, "checkpoints")
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

        self.tracker = MetricsTracker(
            output_dir=exp_dir,
            model_name="rfdetr_nano",
            plot_interval=1,
        )
        self.best_val_loss = float("inf")

    def on_epoch_end(self, log_stats):
        epoch = log_stats.get("epoch", 0)
        train_loss = float(
            log_stats.get("train_loss", log_stats.get("loss", 0.0))
        )
        train_lr = log_stats.get(
            "train_lr", log_stats.get("lr", 0.0)
        )

        if not math.isfinite(train_loss):
            logger.error(
                "NaN/Inf loss detected (%.4f) — requesting stop",
                train_loss,
            )
            self.model_ref.request_early_stop()
            return

        logger.info("  Running evaluation on VAL set...")
        val_t0 = time.time()
        val_metrics = validate_epoch_full(
            model=self.val_model,
            criterion=self.criterion,
            dataset=self.dataset,
            indices=self.val_indices,
            batch_size=self.batch_size,
            num_classes=self.num_classes,
            class_names=self.class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            logger=logger,
            prefix="val",
        )
        logger.info(
            "  Val evaluation completed in %.1fs", time.time() - val_t0
        )

        logger.info("  Running evaluation on TRAIN set...")
        train_eval_t0 = time.time()
        train_eval_metrics = validate_epoch_full(
            model=self.val_model,
            criterion=self.criterion,
            dataset=self.dataset,
            indices=self.train_indices,
            batch_size=self.batch_size,
            num_classes=self.num_classes,
            class_names=self.class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            logger=None,
            prefix="train",
        )
        logger.info(
            "  Train evaluation completed in %.1fs",
            time.time() - train_eval_t0,
        )

        val_loss = val_metrics.get("val_loss", 0.0)
        val_mAP_50 = val_metrics.get("val_mAP_50", 0.0)
        val_mAP_50_95 = val_metrics.get("val_mAP_50_95", 0.0)
        val_precision = val_metrics.get("val_precision", 0.0)
        val_recall = val_metrics.get("val_recall", 0.0)
        val_f1 = val_metrics.get("val_f1", 0.0)
        val_accuracy = val_metrics.get("val_accuracy", 0.0)
        val_num_gt = val_metrics.get("val_num_gt_boxes", 0)
        val_num_pred = val_metrics.get("val_num_pred_boxes", 0)
        val_loss_ce = val_metrics.get("val_loss_ce", 0.0)
        val_loss_bbox = val_metrics.get("val_loss_bbox", 0.0)
        val_loss_giou = val_metrics.get("val_loss_giou", 0.0)

        train_mAP_50 = train_eval_metrics.get("train_mAP_50", 0.0)
        train_mAP_50_95 = train_eval_metrics.get("train_mAP_50_95", 0.0)
        train_precision = train_eval_metrics.get("train_precision", 0.0)
        train_recall = train_eval_metrics.get("train_recall", 0.0)
        train_f1 = train_eval_metrics.get("train_f1", 0.0)
        train_accuracy = train_eval_metrics.get("train_accuracy", 0.0)
        train_num_gt = train_eval_metrics.get("train_num_gt_boxes", 0)
        train_num_pred = train_eval_metrics.get("train_num_pred_boxes", 0)

        lr_val = (
            float(train_lr)
            if isinstance(train_lr, (int, float))
            else 0.0
        )

        logger.info("")
        logger.info("-" * 60)
        logger.info("Epoch %d Summary", epoch)
        logger.info("-" * 60)
        logger.info("  LOSSES:")
        logger.info("    Train Loss (total) : %.4f", train_loss)
        logger.info("    Val Loss   (total) : %.4f", val_loss)
        logger.info("    Val   loss_ce      : %.4f", val_loss_ce)
        logger.info("    Val   loss_bbox    : %.4f", val_loss_bbox)
        logger.info("    Val   loss_giou    : %.4f", val_loss_giou)
        logger.info("  OPTIMIZATION:")
        logger.info("    Learning rate      : %.2e", lr_val)
        logger.info("  TRAIN EVALUATION:")
        logger.info("    mAP@50             : %.4f", train_mAP_50)
        logger.info("    mAP@50:95          : %.4f", train_mAP_50_95)
        logger.info("    Precision          : %.4f", train_precision)
        logger.info("    Recall             : %.4f", train_recall)
        logger.info("    F1 Score           : %.4f", train_f1)
        logger.info("    Accuracy           : %.4f", train_accuracy)
        logger.info("    GT Boxes           : %d", train_num_gt)
        logger.info("    Pred Boxes         : %d", train_num_pred)
        logger.info("  VAL EVALUATION:")
        logger.info("    mAP@50             : %.4f", val_mAP_50)
        logger.info("    mAP@50:95          : %.4f", val_mAP_50_95)
        logger.info("    Precision          : %.4f", val_precision)
        logger.info("    Recall             : %.4f", val_recall)
        logger.info("    F1 Score           : %.4f", val_f1)
        logger.info("    Accuracy           : %.4f", val_accuracy)
        logger.info("    GT Boxes           : %d", val_num_gt)
        logger.info("    Pred Boxes         : %d", val_num_pred)
        logger.info("-" * 60)

        self.tracker.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mAP_50=val_mAP_50,
            val_mAP_50_95=val_mAP_50_95,
            val_precision=val_precision,
            val_recall=val_recall,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            val_num_gt_boxes=val_num_gt,
            val_num_pred_boxes=val_num_pred,
            train_mAP_50=train_mAP_50,
            train_mAP_50_95=train_mAP_50_95,
            train_precision=train_precision,
            train_recall=train_recall,
            train_f1=train_f1,
            train_accuracy=train_accuracy,
            train_num_gt_boxes=train_num_gt,
            train_num_pred_boxes=train_num_pred,
            learning_rate=lr_val,
            per_class_precision=val_metrics.get("per_class_precision"),
            per_class_recall=val_metrics.get("per_class_recall"),
            per_class_f1=val_metrics.get("per_class_f1"),
            per_class_ap50=val_metrics.get("per_class_ap50"),
            val_loss_ce=val_loss_ce,
            val_loss_bbox=val_loss_bbox,
            val_loss_giou=val_loss_giou,
        )

        ckpt_name = (
            f"rfdetr_nano_epoch_{epoch:04d}"
            f"_val_loss_{val_loss:.4f}"
            f"_mAP_{val_mAP_50:.4f}.weights.h5"
        )
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.keras_model.save_weights(ckpt_path)
            logger.info(
                "  [Checkpoint] NEW BEST (val_loss=%.4f): %s",
                val_loss, ckpt_path,
            )
            best_path = os.path.join(
                self.ckpt_dir, "rfdetr_nano_best.weights.h5"
            )
            self.keras_model.save_weights(best_path)
            logger.info("  [Checkpoint] Updated best: %s", best_path)
        else:
            logger.info(
                "  [Checkpoint] No improvement "
                "(current=%.4f, best=%.4f) — skipped",
                val_loss, self.best_val_loss,
            )

        if self.tracker.should_plot(epoch, self.total_epochs):
            self.tracker.generate_plots()
            logger.info(
                "  [Plots] Updated: %s",
                os.path.join(self.exp_dir, "plots"),
            )

    def on_train_end(self):
        self.tracker.generate_plots()
        logger.info("")
        logger.info("=" * 68)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 68)
        logger.info("  Best val loss     : %.4f", self.best_val_loss)
        logger.info("  Experiment dir    : %s", self.exp_dir)
        logger.info("  Checkpoints       : %s", self.ckpt_dir)
        logger.info(
            "  Plots             : %s",
            os.path.join(self.exp_dir, "plots"),
        )
        logger.info("  Metrics log       : %s", self.tracker.log_path)
        if self.tracker.history["epoch"]:
            logger.info("\n%s", self.tracker.format_epoch_summary(-1))
        logger.info("=" * 68)


def main():
    EXP_DIR = _SCRIPT_DIR
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "plots"), exist_ok=True)
    setup_logging(EXP_DIR)

    sys.stdout = _TeeWriter(sys.stdout, os.path.join(EXP_DIR, "output.txt"))

    logger.info("=" * 68)
    logger.info("EXPERIMENT 8: RF-DETR Nano — Fast Fixed-Shape Benchmark")
    logger.info("=" * 68)

    BATCH_SIZE = 4
    EPOCHS = 40
    BASE_LR = 1e-4
    LR_ENCODER = 1.5e-4
    LR_COMPONENT_DECAY = 0.7
    LR_VIT_LAYER_DECAY = 0.8
    WARMUP_EPOCHS = 2.0
    LR_MIN_FACTOR = 0.01
    WEIGHT_DECAY = 1e-4
    CLIP_MAX_NORM = 0.1
    EMA_DECAY = 0.993
    EMA_TAU = 100
    DROP_PATH = 0.1

    logger.info("Loading DeepFish dataset ...")
    ds = DeepFishDataset(resolution=None)
    eval_ds = DeepFishDataset(resolution=384)
    logger.info("DeepFish: %d images, %d classes %s",
                len(ds), ds.num_classes, ds.class_names)

    coco_dir, train_indices, val_indices = prepare_coco_dataset(
        ds, EXP_DIR, val_split=0.2, seed=42,
    )
    logger.info("COCO data: %d train, %d val -> %s",
                len(train_indices), len(val_indices), coco_dir)

    valid_link = os.path.join(coco_dir, "valid")
    val_dir = os.path.join(coco_dir, "val")
    if os.path.isdir(val_dir) and not os.path.exists(valid_link):
        os.symlink(os.path.abspath(val_dir), valid_link)
        logger.info("Created symlink: valid -> val")

    logger.info("Creating RFDETRNano (num_classes=1) ...")
    model = RFDETRNano(num_classes=1)

    _dummy = np.ones((1, 384, 384, 3), dtype="float32") * 0.5
    model.model.model(_dummy, training=True)
    logger.info("Model ready — resolution=%d, group_detr=%d",
                model.model_config.resolution,
                model.model_config.group_detr)

    config = TrainConfig(
        dataset_dir=coco_dir,
        dataset_file="coco_json",
        output_dir=EXP_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=1,
        lr=BASE_LR,
        lr_encoder=LR_ENCODER,
        lr_component_decay=LR_COMPONENT_DECAY,
        lr_vit_layer_decay=LR_VIT_LAYER_DECAY,
        lr_scheduler="cosine",
        lr_min_factor=LR_MIN_FACTOR,
        warmup_epochs=WARMUP_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        clip_max_norm=CLIP_MAX_NORM,
        use_ema=True,
        ema_decay=EMA_DECAY,
        ema_tau=EMA_TAU,
        drop_path=DROP_PATH,
        multi_scale=False,
        expanded_scales=False,
        square_resize_div_64=True,
        amp=True,
        checkpoint_interval=5,
        early_stopping=True,
        early_stopping_patience=15,
        early_stopping_min_delta=0.0005,
        early_stopping_use_ema=True,
        class_names=ds.class_names,
        run_test=False,
        num_workers=2,
    )

    val_criterion, _ = build_criterion_from_config(
        model.model_config, config,
    )

    tracker = ExperimentTracker(
        model_ref=model,
        keras_model=model.model.model,
        criterion=val_criterion,
        dataset=eval_ds,
        train_indices=train_indices,
        val_indices=val_indices,
        num_classes=eval_ds.num_classes,
        class_names=eval_ds.class_names,
        exp_dir=EXP_DIR,
        batch_size=BATCH_SIZE,
        total_epochs=EPOCHS,
        conf_threshold=0.3,
        iou_threshold=0.5,
    )

    model.callbacks["on_fit_epoch_end"].append(tracker.on_epoch_end)
    model.callbacks["on_train_end"].append(tracker.on_train_end)

    exp_config = {
        "experiment": "experiment_8",
        "variant": "RFDETRNano",
        "resolution": model.model_config.resolution,
        "fixed_training_shape": 384,
        "group_detr": model.model_config.group_detr,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": 1,
        "effective_batch_size": BATCH_SIZE,
        "lr": BASE_LR,
        "lr_encoder": LR_ENCODER,
        "lr_component_decay": LR_COMPONENT_DECAY,
        "lr_vit_layer_decay": LR_VIT_LAYER_DECAY,
        "lr_scheduler": "cosine",
        "lr_min_factor": LR_MIN_FACTOR,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "clip_max_norm": CLIP_MAX_NORM,
        "ema_decay": EMA_DECAY,
        "ema_tau": EMA_TAU,
        "drop_path": DROP_PATH,
        "multi_scale": False,
        "expanded_scales": False,
        "num_workers": 2,
        "amp": True,
        "augmentation": "RF-DETR native fixed 384 shape (no engine multi-scale)",
        "evaluation": "pycocotools COCO AP (native train_from_config)",
        "dataset": "DeepFish",
        "train_images": len(train_indices),
        "val_images": len(val_indices),
    }
    config_path = os.path.join(EXP_DIR, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(exp_config, f, indent=2)
    logger.info("Config saved to %s", config_path)

    for k, v in sorted(exp_config.items()):
        logger.info("  %-25s: %s", k, v)

    logger.info("Starting training ...")
    model.train_from_config(config)

    final_path = os.path.join(
        EXP_DIR, "checkpoints", "rfdetr_nano_final.weights.h5",
    )
    model.model.model.save_weights(final_path)
    logger.info("Final weights (EMA-applied) saved -> %s", final_path)


if __name__ == "__main__":
    main()