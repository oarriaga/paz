import math
import time
import datetime
import numpy as np
import keras
from keras import ops

import jax

from paz.models.detection.dino_v2_object_detection.utils.misc import (
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

    Uses a two-phase strategy for JAX compatibility:
      1. **Eager forward + matching** – run the model and the Hungarian
         matcher (which calls scipy) on concrete arrays.
      2. **Traced forward + loss + gradients** – use ``jax.value_and_grad``
         with ``model.stateless_call`` and the pre-computed matching
         indices, so the full loss is JAX-traceable.

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

        # ==================================================================
        # Phase 1 – Eager forward + Hungarian matching
        # ==================================================================
        # The matcher calls scipy (non-JAX-traceable), so we must produce
        # concrete matching indices before entering jax.value_and_grad.
        outputs_eager = model(images, training=False)

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

        # ==================================================================
        # Phase 2 – Traced forward + loss + gradient computation
        # ==================================================================
        trainable_values = [v.value for v in model.trainable_variables]
        non_trainable_values = [v.value for v in model.non_trainable_variables]

        def forward_and_loss(trainable_params):
            """Pure function suitable for ``jax.value_and_grad``."""
            outputs, updated_nt = model.stateless_call(
                trainable_params, non_trainable_values,
                images, training=True,
            )
            total_loss = _compute_loss_with_indices(
                outputs, targets, indices_main, aux_indices,
                criterion, weight_dict, num_boxes_f,
            )
            return total_loss, updated_nt

        grad_fn = jax.value_and_grad(forward_and_loss, has_aux=True)
        (total_loss, updated_nt), grads = grad_fn(trainable_values)

        # ==================================================================
        # Phase 3 – Clip & apply gradients, update state
        # ==================================================================
        if clip_max_norm > 0:
            grads = _clip_grad_norm(grads, clip_max_norm)

        optimizer.apply(grads, model.trainable_variables)

        # Sync non-trainable vars (e.g. BatchNorm running stats)
        for var, val in zip(model.non_trainable_variables, updated_nt):
            var.assign(val)

        # ---- logging -----------------------------------------------------
        loss_value = float(ops.convert_to_numpy(total_loss))
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        # Retrieve current LR from optimizer (for logging only)
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


# ---------------------------------------------------------------------------
# Loss helper (JAX-traceable, no matcher call)
# ---------------------------------------------------------------------------

def _compute_loss_with_indices(
    outputs, targets, indices_main, aux_indices_list,
    criterion, weight_dict, num_boxes_f,
):
    """Compute weighted total loss using pre-computed matching indices.

    This replicates ``SetCriterion.call()`` but **skips the matcher**,
    so the function is safe to call inside ``jax.value_and_grad``.
    """
    num_boxes_t = ops.convert_to_tensor(num_boxes_f, dtype="float32")
    total_loss = ops.convert_to_tensor(0.0, dtype="float32")

    # ---- Main-head losses ------------------------------------------------
    for loss_type in criterion.loss_types:
        l_dict = criterion.get_loss(
            loss_type, outputs, targets, indices_main, num_boxes_t
        )
        for k, v in l_dict.items():
            if k in weight_dict:
                total_loss = total_loss + v * weight_dict[k]

    # ---- Auxiliary decoder-layer losses ----------------------------------
    if "aux_outputs" in outputs:
        for i, aux_out in enumerate(outputs["aux_outputs"]):
            aux_idx = (
                aux_indices_list[i]
                if i < len(aux_indices_list)
                else indices_main
            )
            for loss_type in criterion.loss_types:
                l_dict = criterion.get_loss(
                    loss_type, aux_out, targets, aux_idx, num_boxes_t
                )
                for k, v in l_dict.items():
                    k_aux = f"{k}_{i}"
                    if k_aux in weight_dict:
                        total_loss = total_loss + v * weight_dict[k_aux]

    return total_loss


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
