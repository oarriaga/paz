#!/usr/bin/env python
"""Experiment 4: RF-DETR Nano fine-tuning on DeepFish via the high-level API.

Uses the official RFDETRNano.train() interface with minimal adaptation.
All training logic, optimizer setup, loss computation, and EMA are
handled internally by the RF-DETR framework.

Two targeted fixes are applied via monkey-patch to address Keras-port
bugs in the high-level API:

  1. engine.train_one_epoch Phase 1 uses ``training=False`` which
     produces only 300 queries instead of 300*group_detr=3900.  The
     matcher then crashes because 300 is not divisible by 13.
     Fix: use ``training=True`` in the eager forward pass.

  2. Model.__init__ builds the model with a ``training=False`` dummy
     forward.  Weights load correctly (build() creates all heads), but
     we also do a ``training=True`` forward pass to ensure the JAX
     trace cache is warm for the training-mode path.

Neither the model code nor the engine module is edited on disk.
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
from dataset import DeepFishDataset
from generator import DetectionDataGenerator, prefetch_iterator
from train_utils import prepare_coco_dataset, setup_logging

logger = logging.getLogger(__name__)

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
# Callbacks
# =====================================================================

class HistoryLogger:
    """Accumulates epoch stats and writes structured JSON log."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = []
        self.log_path = os.path.join(output_dir, "history.json")

    def on_epoch_end(self, stats):
        self.history.append(stats)
        # Pretty-print key metrics to console
        epoch = stats.get("epoch", "?")
        loss = stats.get("train_loss", stats.get("loss", "n/a"))
        lr = stats.get("lr", stats.get("train_lr", "n/a"))
        logger.info(
            "Epoch %s  |  loss=%.4f  lr=%s  |  elapsed=%s",
            epoch,
            float(loss) if isinstance(loss, (int, float)) else 0.0,
            f"{float(lr):.2e}" if isinstance(lr, (int, float)) else lr,
            stats.get("epoch_time", "?"),
        )
        # Persist full history
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def on_train_end(self):
        logger.info("Training complete — %d epochs logged to %s",
                     len(self.history), self.log_path)


class BestCheckpointer:
    """Save model weights when training loss improves."""

    def __init__(self, model_ref, output_dir):
        self.model_ref = model_ref
        self.best_loss = float("inf")
        self.best_path = os.path.join(output_dir, "best.weights.h5")

    def on_epoch_end(self, stats):
        loss = float(stats.get("train_loss", stats.get("loss", float("inf"))))
        if loss < self.best_loss:
            self.best_loss = loss
            self.model_ref.model.model.save_weights(self.best_path)
            logger.info("  [Checkpoint] NEW BEST loss=%.4f → %s",
                        loss, self.best_path)


class NanDetector:
    """Stop training immediately if loss becomes NaN/Inf."""

    def __init__(self, model_ref):
        self.model_ref = model_ref

    def on_epoch_end(self, stats):
        loss = stats.get("train_loss", stats.get("loss", 0.0))
        if isinstance(loss, (int, float)) and not math.isfinite(loss):
            logger.error("NaN/Inf loss detected (%.4f) — requesting stop", loss)
            self.model_ref.request_early_stop()


# =====================================================================
# Main
# =====================================================================

def main():
    # ---- Configuration ------------------------------------------------
    EXP_DIR = _SCRIPT_DIR
    os.makedirs(EXP_DIR, exist_ok=True)
    setup_logging(EXP_DIR)

    logger.info("=" * 68)
    logger.info("EXPERIMENT 4: RF-DETR Nano — High-level API — DeepFish")
    logger.info("=" * 68)

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
    # Uses DetectionDataGenerator with pipeline2 augmentation (horizontal
    # flip + brightness/contrast/saturation jitter) plus ImageNet
    # normalization, matching what DINOv2 expects.
    BATCH_SIZE = 16

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
    # Initialise with num_classes=1 (Fish) so the model is built with
    # the correct head size from the start.  Pretrained backbone and
    # transformer weights load via skip_mismatch (only the 91-class
    # classification heads are skipped).  This avoids the
    # reinitialize_detection_head call inside train_from_config, which
    # would discard ALL pretrained weights.
    logger.info("Creating RFDETRNano (num_classes=1) …")
    model = RFDETRNano(num_classes=1)

    # Warm the training-mode trace (model was built with training=False)
    _dummy = np.ones((1, 384, 384, 3), dtype="float32") * 0.5
    model.model.model(_dummy, training=True)

    logger.info("Model ready — resolution=%d, group_detr=%d",
                model.model_config.resolution,
                model.model_config.group_detr)

    # ---- Register callbacks -------------------------------------------
    hist = HistoryLogger(EXP_DIR)
    ckpt = BestCheckpointer(model, EXP_DIR)
    nandet = NanDetector(model)

    model.callbacks["on_fit_epoch_end"].append(hist.on_epoch_end)
    model.callbacks["on_fit_epoch_end"].append(ckpt.on_epoch_end)
    model.callbacks["on_fit_epoch_end"].append(nandet.on_epoch_end)
    model.callbacks["on_train_end"].append(hist.on_train_end)

    # ---- Train --------------------------------------------------------
    # All parameters below are RF-DETR defaults (from TrainConfig).
    # The only overrides are dataset_dir, output_dir, epochs, and
    # batch_size — everything else uses the framework's built-in values.
    logger.info("Starting training …")
    config = TrainConfig(
        dataset_dir=coco_dir,
        output_dir=EXP_DIR,
        epochs=20,
        batch_size=BATCH_SIZE,
        grad_accum_steps=1,
        lr=1e-4,
        lr_encoder=1.5e-4,
        weight_decay=1e-4,
        clip_max_norm=0.1,
        use_ema=True,
        ema_decay=0.993,
        ema_tau=100,
        lr_vit_layer_decay=0.8,
        lr_component_decay=0.7,
        warmup_epochs=0.0,
        checkpoint_interval=10,
        early_stopping=True,
    )
    model.train_from_config(config, data_loader_train=train_loader)

    # ---- Save final model ---------------------------------------------
    final_path = os.path.join(EXP_DIR, "final.weights.h5")
    model.model.model.save_weights(final_path)
    logger.info("Final weights (EMA-applied) saved → %s", final_path)


if __name__ == "__main__":
    main()
