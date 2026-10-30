#!/usr/bin/env python
"""Experiment 5: RF-DETR Nano fine-tuning on DeepFish via the high-level API.

Uses the official RFDETRNano.train() interface with minimal adaptation.
All training logic, optimizer setup, loss computation, and EMA are
handled internally by the RF-DETR framework.

Three targeted fixes are applied via monkey-patch to address Keras-port
bugs / gaps in the high-level API:

  1. engine.train_one_epoch Phase 1 uses ``training=False`` which
     produces only 300 queries instead of 300*group_detr=3900.  The
     matcher then crashes because 300 is not divisible by 13.
     Fix: use ``training=True`` in the eager forward pass.

  2. Model.__init__ builds the model with a ``training=False`` dummy
     forward.  Weights load correctly (build() creates all heads), but
     we also do a ``training=True`` forward pass to ensure the JAX
     trace cache is warm for the training-mode path.

  3. train_from_config creates a flat-LR AdamW optimizer, ignoring the
     ``build_lr_lambda`` / ``LambdaLRSchedule`` that engine.py provides.
     Fix: temporarily wrap ``keras.optimizers.AdamW`` so the constructor
     replaces the scalar LR with a cosine ``LambdaLRSchedule``.
     Base LR = 1e-4, cosine annealing to 0.

Additionally, stdout is tee'd to the log file so that per-step progress
from ``MetricLogger.log_every`` (which uses ``print()``) also appears
in ``output.txt``.

Neither the model code nor the engine module is edited on disk.

Experiment tracking (validation, checkpoints, plots, epoch summaries)
is implemented via ``on_fit_epoch_end`` / ``on_train_end`` callbacks,
matching the behaviour of Experiments 1–3.
"""

import os
import sys
import json
import math
import time
import datetime
import logging

import numpy as np


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

    # Forward attribute lookups (fileno, encoding, etc.) to the original
    def __getattr__(self, name):
        return getattr(self._original, name)

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
# Monkey-patch: fix engine.train_one_epoch for group_detr > 1
# ---------------------------------------------------------------------------
# The Keras port's eager forward uses training=False, producing only
# num_queries (300) outputs.  The Hungarian matcher then tries to split
# 300 by group_detr=13, which fails because 300 % 13 != 0.
# Fix: use training=True in Phase 1 so all 3900 outputs are produced.

import keras
from keras import ops
import jax

import paz.models.detection.dino_v2_object_detection.engine as _engine
from paz.models.detection.dino_v2_object_detection.utils.misc import (
    MetricLogger,
    SmoothedValue,
)


def _patched_train_one_epoch(
    model, criterion, optimizer, data_iterator, num_steps, epoch,
    clip_max_norm=0.1, print_freq=10,
):
    """Patched ``train_one_epoch`` — uses ``training=True`` in Phase 1.

    Only change vs. original: line marked [PATCHED].
    """
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    weight_dict = criterion.weight_dict
    group_detr = criterion.group_detr
    sum_group_losses = getattr(criterion, "sum_group_losses", False)

    start_time = time.time()
    for step, (images, targets) in enumerate(
        metric_logger.log_every(data_iterator, print_freq, header)
    ):
        images = ops.convert_to_tensor(images, dtype="float32")

        # Phase 1 — Eager forward + Hungarian matching
        outputs_eager = model(images, training=True)          # [PATCHED]

        outputs_for_match = {
            k: v for k, v in outputs_eager.items() if k != "aux_outputs"
        }
        indices_main = criterion.matcher(
            outputs_for_match, targets, group_detr=group_detr
        )

        aux_indices = []
        if "aux_outputs" in outputs_eager:
            for aux_out in outputs_eager["aux_outputs"]:
                aux_indices.append(
                    criterion.matcher(
                        aux_out, targets, group_detr=group_detr
                    )
                )

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes_f = max(float(num_boxes), 1.0)

        # Phase 2 — Traced forward + loss + gradient computation
        trainable_values = [v.value for v in model.trainable_variables]
        non_trainable_values = [v.value for v in model.non_trainable_variables]

        def forward_and_loss(trainable_params):
            outputs, updated_nt = model.stateless_call(
                trainable_params, non_trainable_values,
                images, training=True,
            )
            total_loss = _engine._compute_loss_with_indices(
                outputs, targets, indices_main, aux_indices,
                criterion, weight_dict, num_boxes_f,
            )
            return total_loss, updated_nt

        grad_fn = jax.value_and_grad(forward_and_loss, has_aux=True)
        (total_loss, updated_nt), grads = grad_fn(trainable_values)

        # Phase 3 — Clip & apply gradients, update state
        if clip_max_norm > 0:
            grads = _engine._clip_grad_norm(grads, clip_max_norm)

        optimizer.apply(grads, model.trainable_variables)

        for var, val in zip(model.non_trainable_variables, updated_nt):
            var.assign(val)

        loss_value = float(ops.convert_to_numpy(total_loss))
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if hasattr(optimizer, "learning_rate"):
            lr_val = optimizer.learning_rate
            if callable(lr_val):
                lr_val = float(lr_val(optimizer.iterations))
            else:
                lr_val = float(lr_val)
        else:
            lr_val = 0.0
        metric_logger.update(loss=loss_value, lr=lr_val)

        if step >= num_steps - 1:
            break

    elapsed = time.time() - start_time
    print(
        f"{header} Total time: {datetime.timedelta(seconds=int(elapsed))} "
        f"({elapsed / max(1, num_steps):.4f} s / it)"
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Apply the patch before any training code imports the function
_engine.train_one_epoch = _patched_train_one_epoch

# ---------------------------------------------------------------------------
# Now import the high-level API and dataset utilities
# ---------------------------------------------------------------------------
from paz.models.detection.dino_v2_object_detection.detr import RFDETRNano
from paz.models.detection.dino_v2_object_detection.config import TrainConfig
from paz.models.detection.dino_v2_object_detection.main import (
    build_criterion_from_config,
)
from paz.models.detection.dino_v2_object_detection.engine import (
    build_lr_lambda,
    LambdaLRSchedule,
)
from dataset import DeepFishDataset
from generator import DetectionDataGenerator, prefetch_iterator
from train_utils import prepare_coco_dataset, setup_logging, validate_epoch_full
from metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkey-patch 3: inject cosine LR schedule into the optimizer
# ---------------------------------------------------------------------------
# train_from_config creates:
#   optimizer = keras.optimizers.AdamW(learning_rate=config.lr, ...)
# which is a flat scalar.  engine.py provides build_lr_lambda /
# LambdaLRSchedule (cosine with optional warmup) but train_from_config
# never calls them.  We wrap AdamW so the constructor replaces the
# scalar LR with a LambdaLRSchedule.  The wrapper is activated just
# before train_from_config and deactivated after.

_OriginalAdamW = keras.optimizers.AdamW
_LR_SCHEDULE_CONFIG = {}   # populated by main() before train_from_config


class _ScheduledAdamW(_OriginalAdamW):
    """AdamW that swaps a scalar LR for a cosine LambdaLRSchedule."""

    def __init__(self, learning_rate=0.001, **kwargs):
        cfg = _LR_SCHEDULE_CONFIG
        if cfg and isinstance(learning_rate, (int, float)):
            lr_lambda = build_lr_lambda(
                num_training_steps_per_epoch=cfg["steps_per_epoch"],
                epochs=cfg["epochs"],
                warmup_epochs=cfg["warmup_epochs"],
                lr_scheduler="cosine",
                lr_min_factor=cfg.get("lr_min_factor", 0.0),
            )
            learning_rate = LambdaLRSchedule(
                base_lr=cfg["base_lr"], lr_lambda=lr_lambda,
            )
            logger.info(
                "  [LR Schedule] Cosine schedule injected: "
                "base_lr=%.1e, warmup=%s epochs, %d steps/epoch",
                cfg["base_lr"], cfg["warmup_epochs"],
                cfg["steps_per_epoch"],
            )
        super().__init__(learning_rate=learning_rate, **kwargs)


# ImageNet channel statistics (DINOv2 pretraining distribution)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _AugmentedDataLoader:
    """Wraps DetectionDataGenerator to add ImageNet normalization.

    Compatible with ``train_from_config`` — supports ``__len__`` and
    ``__iter__`` yielding ``(images_np, targets)`` tuples.

    Automatically calls ``set_epoch`` on the underlying generator each
    time ``__iter__`` is invoked, so per-epoch reshuffling works even
    though ``train_from_config`` doesn't call ``set_epoch`` itself.
    """

    def __init__(self, generator, max_prefetch=4):
        self._generator = generator
        self._max_prefetch = max_prefetch
        self._epoch = 0

    def __len__(self):
        return len(self._generator)

    def __iter__(self):
        self._generator.set_epoch(self._epoch)
        self._epoch += 1
        for images_np, targets in prefetch_iterator(
            self._generator, max_prefetch=self._max_prefetch
        ):
            # images_np: (B, H, W, 3) float32 in [0, 1] after augmentation
            # Apply ImageNet normalization (same as _COCODataLoader)
            images_np = (images_np - _IMAGENET_MEAN) / _IMAGENET_STD
            yield images_np, targets


# =====================================================================
# Experiment tracker — validation, checkpoints, plots, summaries
# =====================================================================

class ExperimentTracker:
    """Full experiment tracking via callbacks, matching Experiments 1–3.

    Plugs into the high-level API via ``on_fit_epoch_end`` /
    ``on_train_end`` callbacks and provides:

    - Per-epoch validation (val + train-eval) using ``validate_epoch_full``
    - Structured epoch summaries with timestamps (via ``logger.info``)
    - Checkpoint naming: ``rfdetr_nano_epoch_EEEE_val_loss_V.VVVV_mAP_M.MMMM.weights.h5``
    - Best checkpoint: ``rfdetr_nano_best.weights.h5``
    - Plot generation via ``MetricsTracker``
    - NaN/Inf detection with early-stop request
    """

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

    # ------------------------------------------------------------------

    def on_epoch_end(self, log_stats):
        epoch = log_stats.get("epoch", 0)
        train_loss = float(
            log_stats.get("train_loss", log_stats.get("loss", 0.0))
        )
        train_lr = log_stats.get(
            "train_lr", log_stats.get("lr", 0.0)
        )

        # ---- NaN / Inf guard -----------------------------------------
        if not math.isfinite(train_loss):
            logger.error(
                "NaN/Inf loss detected (%.4f) — requesting stop",
                train_loss,
            )
            self.model_ref.request_early_stop()
            return

        # ---- Validation on VAL set -----------------------------------
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

        # ---- Validation on TRAIN set (monitor overfitting) -----------
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

        # ---- Extract all metrics -------------------------------------
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

        # ---- Per-epoch summary (matches Experiments 1–3) -------------
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

        # ---- MetricsTracker ------------------------------------------
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

        # ---- Checkpointing (best_keep strategy) ----------------------
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

        # ---- Plots ---------------------------------------------------
        if self.tracker.should_plot(epoch, self.total_epochs):
            self.tracker.generate_plots()
            logger.info(
                "  [Plots] Updated: %s",
                os.path.join(self.exp_dir, "plots"),
            )

    # ------------------------------------------------------------------

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


# =====================================================================
# Main
# =====================================================================

def main():
    # ---- Configuration ------------------------------------------------
    EXP_DIR = _SCRIPT_DIR
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(EXP_DIR, "plots"), exist_ok=True)
    setup_logging(EXP_DIR)

    # Tee stdout so MetricLogger per-step prints appear in output.txt
    sys.stdout = _TeeWriter(sys.stdout, os.path.join(EXP_DIR, "output.txt"))

    logger.info("=" * 68)
    logger.info("EXPERIMENT 5: RF-DETR Nano — High-level API — DeepFish")
    logger.info("=" * 68)

    BATCH_SIZE = 16
    EPOCHS = 20
    BASE_LR = 1e-4
    WARMUP_EPOCHS = 0.0

    # ---- Prepare DeepFish in COCO format ------------------------------
    logger.info("Loading DeepFish dataset …")
    ds = DeepFishDataset(resolution=384)
    logger.info("DeepFish: %d images, %d classes %s",
                len(ds), ds.num_classes, ds.class_names)

    coco_dir, train_indices, val_indices = prepare_coco_dataset(
        ds, EXP_DIR, val_split=0.2, seed=42,
    )
    logger.info("COCO data: %d train, %d val → %s",
                len(train_indices), len(val_indices), coco_dir)

    # ---- Build augmented data loader ----------------------------------
    train_gen = DetectionDataGenerator(
        dataset=ds,
        indices=train_indices,
        batch_size=BATCH_SIZE,
        augmentation="pipeline2",
        seed=42,
        shuffle=True,
    )
    train_loader = _AugmentedDataLoader(train_gen, max_prefetch=4)
    logger.info("Train loader: %d batches of %d (pipeline2 + ImageNet norm)",
                len(train_loader), BATCH_SIZE)

    # ---- Create model -------------------------------------------------
    logger.info("Creating RFDETRNano (num_classes=1) …")
    model = RFDETRNano(num_classes=1)

    # Warm the training-mode trace (model was built with training=False)
    _dummy = np.ones((1, 384, 384, 3), dtype="float32") * 0.5
    model.model.model(_dummy, training=True)

    logger.info("Model ready — resolution=%d, group_detr=%d",
                model.model_config.resolution,
                model.model_config.group_detr)

    # ---- Build criterion for validation -------------------------------
    config = TrainConfig(
        dataset_dir=coco_dir,
        output_dir=EXP_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=1,
        lr=BASE_LR,
        lr_encoder=1.5e-4,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        lr_vit_layer_decay=0.8,
        lr_component_decay=0.7,
        warmup_epochs=WARMUP_EPOCHS,
        checkpoint_interval=10,
        early_stopping=True,
    )

    val_criterion, _ = build_criterion_from_config(
        model.model_config, config
    )

    # ---- Register experiment tracker ----------------------------------
    tracker = ExperimentTracker(
        model_ref=model,
        keras_model=model.model.model,
        criterion=val_criterion,
        dataset=ds,
        train_indices=train_indices,
        val_indices=val_indices,
        num_classes=ds.num_classes,
        class_names=ds.class_names,
        exp_dir=EXP_DIR,
        batch_size=BATCH_SIZE,
        total_epochs=EPOCHS,
        conf_threshold=0.3,
        iou_threshold=0.5,
    )

    model.callbacks["on_fit_epoch_end"].append(tracker.on_epoch_end)
    model.callbacks["on_train_end"].append(tracker.on_train_end)

    # ---- Activate LR schedule monkey-patch ----------------------------
    _LR_SCHEDULE_CONFIG.update({
        "base_lr": BASE_LR,
        "steps_per_epoch": len(train_loader),
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "lr_min_factor": 0.0,
    })
    keras.optimizers.AdamW = _ScheduledAdamW

    # ---- Train --------------------------------------------------------
    logger.info("Starting training …")
    try:
        model.train_from_config(config, data_loader_train=train_loader)
    finally:
        keras.optimizers.AdamW = _OriginalAdamW  # restore

    # ---- Save final model ---------------------------------------------
    final_path = os.path.join(EXP_DIR, "checkpoints", "rfdetr_nano_final.weights.h5")
    model.model.model.save_weights(final_path)
    logger.info("Final weights (EMA-applied) saved → %s", final_path)


if __name__ == "__main__":
    main()
