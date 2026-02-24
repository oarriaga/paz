import math
import time
import datetime
import numpy as np
import keras
from keras import ops

from examples.dino_v2_object_detection.utils.misc import (
    MetricLogger,
    SmoothedValue,
)


# ---------------------------------------------------------------------------
# Learning-rate schedule helpers
# ---------------------------------------------------------------------------


def build_lr_lambda(
    num_training_steps_per_epoch,
    epochs,
    warmup_epochs,
    lr_scheduler="step",
    lr_drop=100,
    lr_min_factor=0.0,
):
    """Return a callable ``lr_lambda(step) -> float`` for LambdaLR-style use.

    Mirrors the PyTorch ``lr_lambda`` closure in ``main.py``.
    """
    total_steps = num_training_steps_per_epoch * epochs
    warmup_steps = int(num_training_steps_per_epoch * warmup_epochs)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if lr_scheduler == "cosine":
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return lr_min_factor + (1 - lr_min_factor) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:  # step
            drop_step = lr_drop * num_training_steps_per_epoch
            return 1.0 if current_step < drop_step else 0.1

    return lr_lambda


# ---------------------------------------------------------------------------
# Simple Keras LR scheduler wrapping lr_lambda
# ---------------------------------------------------------------------------


class LambdaLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Keras LR schedule that delegates to a ``lr_lambda`` function."""

    def __init__(self, base_lr, lr_lambda):
        super().__init__()
        self.base_lr = base_lr
        self.lr_lambda = lr_lambda

    def __call__(self, step):
        return self.base_lr * self.lr_lambda(int(step))

    def get_config(self):
        return {"base_lr": self.base_lr}


# ---------------------------------------------------------------------------
# Training loop for one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_iterator,
    num_steps,
    epoch,
    clip_max_norm=0.1,
    print_freq=10,
):
    """Train *model* for one epoch using *data_iterator*.

    Parameters
    ----------
    model : keras.Model
        The LWDETR Keras model.
    criterion : SetCriterion (Keras layer)
        Computes losses given outputs and targets.
    optimizer : keras.optimizers.Optimizer
        Must already be compiled with the model.
    data_iterator : iterable
        Yields ``(images, targets)`` batches.  ``images`` is a
        ``(B, H, W, 3)`` Keras tensor; ``targets`` is a list of dicts
        with ``"labels"`` and ``"boxes"`` keys.
    num_steps : int
        Total steps in this epoch (used for logging only).
    epoch : int
        Current epoch index.
    clip_max_norm : float
        Max gradient norm for clipping (0 disables).
    print_freq : int
        Print every N steps.

    Returns
    -------
    dict : averaged metric values.
    """
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    start_time = time.time()
    for step, (images, targets) in enumerate(
        metric_logger.log_every(data_iterator, print_freq, header)
    ):
        with keras.utils.jax_scope():
            # Forward
            outputs = model(images, training=True)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            total_loss = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict
            )

        # Compute gradients
        trainable_vars = model.trainable_variables
        grads = keras.backend.gradients(total_loss, trainable_vars)

        if clip_max_norm > 0:
            grads = _clip_grad_norm(grads, clip_max_norm)

        optimizer.apply(grads, trainable_vars)

        loss_value = float(ops.convert_to_numpy(total_loss))
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # Logging
        loss_dict_np = {k: float(ops.convert_to_numpy(v)) for k, v in loss_dict.items()}
        loss_dict_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_np.items() if k in weight_dict
        }
        metric_logger.update(loss=loss_value, **loss_dict_scaled)

        if step >= num_steps - 1:
            break

    elapsed = time.time() - start_time
    print(
        f"{header} Total time: {datetime.timedelta(seconds=int(elapsed))} "
        f"({elapsed / max(1, num_steps):.4f} s / it)"
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _clip_grad_norm(grads, max_norm):
    """Global gradient clipping (mirrors ``torch.nn.utils.clip_grad_norm_``)."""
    total_norm_sq = sum(ops.sum(g**2) for g in grads if g is not None)
    total_norm = ops.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = ops.minimum(clip_coef, 1.0)
    return [g * clip_coef if g is not None else g for g in grads]


# ---------------------------------------------------------------------------
# Evaluation (forward-only, no COCO eval)
# ---------------------------------------------------------------------------


def evaluate(model, data_iterator, num_steps=None, print_freq=10):
    """Run the model in evaluation mode and collect predictions.

    Parameters
    ----------
    model : keras.Model
    data_iterator : iterable
        Yields ``(images, targets)`` batches.
    num_steps : int or None
        Max steps; if ``None``, iterate until exhaustion.
    print_freq : int

    Returns
    -------
    list[dict] : one dict per image with ``pred_logits``, ``pred_boxes``
        (and optionally ``pred_masks``).
    """
    all_results = []
    for step, (images, _targets) in enumerate(data_iterator):
        outputs = model(images, training=False)
        # Convert to numpy for downstream consumption
        result = {
            "pred_logits": ops.convert_to_numpy(outputs["pred_logits"]),
            "pred_boxes": ops.convert_to_numpy(outputs["pred_boxes"]),
        }
        if "pred_masks" in outputs:
            result["pred_masks"] = ops.convert_to_numpy(outputs["pred_masks"])
        all_results.append(result)

        if num_steps is not None and step >= num_steps - 1:
            break

    return all_results


# ---------------------------------------------------------------------------
# Metric helpers (no COCO dependency)
# ---------------------------------------------------------------------------


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
    """Sweep confidence thresholds and compute P/R/F1 at each.

    This is a pure-NumPy replica of the PyTorch ``sweep_confidence_thresholds``.
    """
    results = []
    for conf_thresh in conf_thresholds:
        per_class_prec, per_class_rec, per_class_f1 = [], [], []
        for k in range(len(per_class_data)):
            data = per_class_data[k]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            above = scores >= conf_thresh
            valid = above & ~ignore
            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_prec.append(prec)
            per_class_rec.append(rec)
            per_class_f1.append(f1)

        if classes_with_gt:
            macro_p = np.mean([per_class_prec[k] for k in classes_with_gt])
            macro_r = np.mean([per_class_rec[k] for k in classes_with_gt])
            macro_f1 = np.mean([per_class_f1[k] for k in classes_with_gt])
        else:
            macro_p = macro_r = macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "per_class_prec": np.array(per_class_prec),
                "per_class_rec": np.array(per_class_rec),
            }
        )
    return results
