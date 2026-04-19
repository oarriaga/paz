import math
import random
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
    """Return a callable ``lr_lambda(step) -> float`` for use with
    ``LambdaLRSchedule``.

    Supports ``'step'`` (drop at ``lr_drop`` epochs) and ``'cosine'``
    (cosine annealing with optional warm-up) schedules.

    Args:
        num_training_steps_per_epoch (int): Steps in one epoch.
        epochs (int): Total training epochs.
        warmup_epochs (float): Linear warm-up duration in epochs.
        lr_scheduler (str): ``'step'`` or ``'cosine'``.
        lr_drop (int): Epoch to drop LR (step schedule only).
        lr_min_factor (float): Minimum LR as a fraction of base LR.

    Returns:
        callable: ``lr_lambda(current_step) -> float`` multiplier.
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
# Simple LR scheduler wrapping lr_lambda
# ---------------------------------------------------------------------------


class LambdaLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """LR schedule that delegates to a ``lr_lambda`` callable.

    Multiplies ``base_lr`` by ``lr_lambda(step)`` at each training step.

    Attributes:
        base_lr (float): Base learning rate.
        lr_lambda (callable): Step-to-multiplier function.
    """

    def __init__(self, base_lr, lr_lambda):
        super().__init__()
        self.base_lr = base_lr
        self.lr_lambda = lr_lambda

    def __call__(self, step):
        return self.base_lr * self.lr_lambda(int(step))

    def get_config(self):
        return {"base_lr": self.base_lr}


# ---------------------------------------------------------------------------
# Drop path / dropout schedule builder
# ---------------------------------------------------------------------------


def build_drop_schedule(
    drop_rate,
    epochs,
    niter_per_ep,
    cutoff_epoch=0,
    mode="standard",
    schedule="constant",
):
    """Pre-compute a per-step drop-rate array.

    Args:
        drop_rate (float): Target drop rate.
        epochs (int): Total training epochs.
        niter_per_ep (int): Steps per epoch.
        cutoff_epoch (int): Epoch boundary for early/late modes.
        mode (str): ``'standard'``, ``'early'``, or ``'late'``.
        schedule (str): ``'constant'`` or ``'linear'`` (early mode only).

    Returns:
        np.ndarray: Float32 array of length ``epochs * niter_per_ep``.
    """
    assert mode in ("standard", "early", "late")
    if mode == "standard":
        return np.full(epochs * niter_per_ep, drop_rate, dtype="float32")

    early_iters = cutoff_epoch * niter_per_ep
    late_iters = (epochs - cutoff_epoch) * niter_per_ep

    if mode == "early":
        assert schedule in ("constant", "linear")
        if schedule == "constant":
            early_schedule = np.full(early_iters, drop_rate, dtype="float32")
        elif schedule == "linear":
            early_schedule = np.linspace(
                drop_rate, 0, early_iters, dtype="float32"
            )
        final_schedule = np.concatenate(
            (early_schedule, np.full(late_iters, 0, dtype="float32"))
        )
    elif mode == "late":
        assert schedule in ("constant",)
        early_schedule = np.full(early_iters, 0, dtype="float32")
        final_schedule = np.concatenate(
            (early_schedule, np.full(late_iters, drop_rate, dtype="float32"))
        )

    assert len(final_schedule) == epochs * niter_per_ep
    return final_schedule


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
    lr_multipliers=None,
    ema_m=None,
    grad_accum_steps=1,
    multi_scale_config=None,
    drop_path_schedule=None,
    dropout_schedule=None,
    vit_encoder_num_layers=None,
    amp=False,
):
    """Train *model* for one epoch using *data_iterator*.

    Uses a two-phase strategy for JAX compatibility:
      1. **Eager forward + matching** -- run the model and the Hungarian
         matcher (which calls scipy) on concrete arrays.
      2. **Traced forward + loss + gradients** -- use ``jax.value_and_grad``
         with ``model.stateless_call`` and the pre-computed matching
         indices, so the full loss computation is JAX-traceable.

    Args:
        model: The LWDETR model.
        criterion (SetCriterion): Computes losses given outputs and targets.
        optimizer: Optimiser instance.
        data_iterator (iterable): Yields ``(images, targets)`` batches.
            ``images`` is ``(B, H, W, 3)`` float32; ``targets`` is a list
            of dicts with ``labels`` and ``boxes`` keys.
        num_steps (int): Total steps in this epoch (for logging).
        epoch (int): Current epoch index.
        clip_max_norm (float): Max gradient norm for clipping (0 disables).
        print_freq (int): Print every N steps.
        lr_multipliers (dict or None): Per-variable gradient multipliers
            for differential learning rates.
        ema_m: Optional EMA wrapper; ``ema_m.update(model)`` is called
            after every optimiser step.
        grad_accum_steps (int): Number of sub-batches to accumulate
            gradients over before each optimiser step.
        multi_scale_config (dict or None): When not None, enables
            per-batch multi-scale resizing.  Must contain a ``scales``
            list of valid square sizes.
        drop_path_schedule (np.ndarray or None): Pre-computed per-step
            drop-path rates.  When not None, the model's backbone
            drop-path rate is updated at each iteration.
        dropout_schedule (np.ndarray or None): Pre-computed per-step
            dropout rates.  When not None, all Dropout layers in the
            transformer are updated at each iteration.
        vit_encoder_num_layers (int or None): Number of ViT encoder
            layers (needed for per-layer linear scaling of drop path).

    Returns:
        dict: Averaged metric values across the epoch.
    """
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"

    weight_dict = criterion.weight_dict
    group_detr = criterion.group_detr
    sum_group_losses = getattr(criterion, "sum_group_losses", False)

    ms_scales = (
        multi_scale_config["scales"] if multi_scale_config is not None
        else None
    )

    start_time = time.time()
    for step, (images, targets) in enumerate(
        metric_logger.log_every(data_iterator, print_freq, header)
    ):
        global_step = epoch * num_steps + step

        # ---- Per-step drop path scheduling ----
        if drop_path_schedule is not None:
            if global_step < len(drop_path_schedule):
                dp_rate = float(drop_path_schedule[global_step])
                if hasattr(model, "update_drop_path"):
                    model.update_drop_path(
                        dp_rate, vit_encoder_num_layers
                    )

        # ---- Per-step dropout scheduling ----
        if dropout_schedule is not None:
            if global_step < len(dropout_schedule):
                do_rate = float(dropout_schedule[global_step])
                if hasattr(model, "update_dropout"):
                    model.update_dropout(do_rate)

        # Unpack images (may be a plain array or a (tensor, mask) tuple)
        if isinstance(images, (list, tuple)) and len(images) == 2:
            img_tensor, img_mask = images
            images = ops.convert_to_tensor(img_tensor, dtype="float32")
            img_mask = ops.convert_to_tensor(img_mask, dtype="bool")
        else:
            images = ops.convert_to_tensor(images, dtype="float32")
            img_mask = None

        # ---- Per-batch multi-scale resize (deterministic per step) ----
        if ms_scales is not None:
            random.seed(step)
            scale = random.choice(ms_scales)
            images = ops.image.resize(images, (scale, scale))
            if img_mask is not None:
                img_mask = ops.cast(
                    ops.image.resize(
                        ops.cast(img_mask[:, :, :, None], "float32"),
                        (scale, scale),
                        interpolation="nearest",
                    )[:, :, :, 0],
                    "bool",
                )

        batch_size = int(images.shape[0])
        sub_batch_size = batch_size // grad_accum_steps

        accumulated_grads = None
        accumulated_loss = 0.0
        last_updated_nt = None

        for accum_i in range(grad_accum_steps):
            start_idx = accum_i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            sub_images = images[start_idx:final_idx]
            sub_targets = targets[start_idx:final_idx]

            # Build model input (with mask if available)
            if img_mask is not None:
                sub_mask = img_mask[start_idx:final_idx]
                model_input = (sub_images, sub_mask)
            else:
                model_input = sub_images

            # ==============================================================
            # Phase 1 – Eager forward + Hungarian matching
            # ==============================================================
            outputs_eager = model(model_input, training=False)

            outputs_for_match = {
                k: v for k, v in outputs_eager.items()
                if k not in ("aux_outputs", "enc_outputs")
            }
            indices_main = criterion.matcher(
                outputs_for_match, sub_targets, group_detr=group_detr
            )

            aux_indices = []
            if "aux_outputs" in outputs_eager:
                for aux_out in outputs_eager["aux_outputs"]:
                    aux_indices.append(
                        criterion.matcher(
                            aux_out, sub_targets, group_detr=group_detr
                        )
                    )

            enc_indices = None
            if "enc_outputs" in outputs_eager:
                enc_indices = criterion.matcher(
                    outputs_eager["enc_outputs"], sub_targets,
                    group_detr=group_detr,
                )

            num_boxes = sum(len(t["labels"]) for t in sub_targets)
            if not sum_group_losses:
                num_boxes = num_boxes * group_detr
            num_boxes_f = max(float(num_boxes), 1.0)

            # ==============================================================
            # Phase 2 – Traced forward + loss + gradient computation
            # ==============================================================
            trainable_values = [v.value for v in model.trainable_variables]
            non_trainable_values = [
                v.value for v in model.non_trainable_variables
            ]

            # Cast inputs to bfloat16 when AMP is enabled
            fwd_images = sub_images
            if amp:
                fwd_images = ops.cast(sub_images, "bfloat16")

            # Build traced input (with or without mask)
            if img_mask is not None:
                fwd_input = (fwd_images, sub_mask)
            else:
                fwd_input = fwd_images

            def forward_and_loss(trainable_params):
                """Pure function suitable for ``jax.value_and_grad``."""
                outputs, updated_nt = model.stateless_call(
                    trainable_params, non_trainable_values,
                    fwd_input, training=True,
                )
                # Cast outputs to float32 for loss computation
                if amp:
                    outputs = {
                        k: (ops.cast(v, "float32")
                             if hasattr(v, "dtype") else v)
                        for k, v in outputs.items()
                    }
                total_loss = _compute_loss_with_indices(
                    outputs, sub_targets, indices_main, aux_indices,
                    enc_indices, criterion, weight_dict, num_boxes_f,
                )
                return total_loss, updated_nt

            grad_fn = jax.value_and_grad(forward_and_loss, has_aux=True)
            (sub_loss, updated_nt), sub_grads = grad_fn(trainable_values)

            # Cast gradients to float32 when AMP is enabled
            if amp:
                sub_grads = [ops.cast(g, "float32") for g in sub_grads]

            # Scale gradients by 1/grad_accum_steps
            scale = 1.0 / grad_accum_steps
            sub_grads = [g * scale for g in sub_grads]
            accumulated_loss += float(ops.convert_to_numpy(sub_loss)) * scale
            last_updated_nt = updated_nt

            if accumulated_grads is None:
                accumulated_grads = sub_grads
            else:
                accumulated_grads = [
                    a + g for a, g in zip(accumulated_grads, sub_grads)
                ]

        grads = accumulated_grads

        # ==================================================================
        # Phase 3 – Check for NaN/Inf gradients, clip & apply
        # ==================================================================

        # Detect gradient overflow — skip the optimiser step entirely
        # (similar to GradScaler behaviour in mixed-precision training)
        if _has_nan_or_inf(grads):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "NaN/Inf gradients at step %d — skipping update", step,
            )
            metric_logger.update(loss=accumulated_loss, lr=0.0,
                                 grad_overflow=1.0)
            if step >= num_steps - 1:
                break
            continue

        if clip_max_norm > 0:
            grads = _clip_grad_norm(grads, clip_max_norm)

        if lr_multipliers is not None:
            grads = [
                g * lr_multipliers.get(v.path, 1.0)
                for g, v in zip(grads, model.trainable_variables)
            ]

        optimizer.apply(grads, model.trainable_variables)

        # Sync non-trainable vars (e.g. BatchNorm running stats)
        for var, val in zip(model.non_trainable_variables, last_updated_nt):
            var.assign(val)

        # Per-step EMA update
        if ema_m is not None:
            ema_m.update(model)

        # ---- logging -----------------------------------------------------
        loss_value = accumulated_loss
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
    enc_indices, criterion, weight_dict, num_boxes_f,
):
    """Compute weighted total loss using pre-computed matching indices.

    Replicates the ``SetCriterion.call()`` logic but **skips the matcher**,
    so the function is safe to call inside ``jax.value_and_grad``.

    Args:
        outputs (dict): Model outputs with ``pred_logits``, ``pred_boxes``,
            and optionally ``aux_outputs`` and ``enc_outputs``.
        targets (list[dict]): Ground-truth targets.
        indices_main (list): Main-head matching indices.
        aux_indices_list (list): Per-layer auxiliary matching indices.
        enc_indices (list or None): Encoder output matching indices.
        criterion (SetCriterion): Loss module.
        weight_dict (dict): Loss name → weight mapping.
        num_boxes_f (float): Total matched boxes (for normalisation).

    Returns:
        Tensor: Scalar weighted total loss.
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

    # ---- Two-stage encoder output losses ---------------------------------
    if "enc_outputs" in outputs and enc_indices is not None:
        enc_out = outputs["enc_outputs"]
        for loss_type in criterion.loss_types:
            kwargs = {}
            if loss_type == "labels":
                kwargs["log"] = False
            l_dict = criterion.get_loss(
                loss_type, enc_out, targets, enc_indices, num_boxes_t,
                **kwargs,
            )
            for k, v in l_dict.items():
                k_enc = f"{k}_enc"
                if k_enc in weight_dict:
                    total_loss = total_loss + v * weight_dict[k_enc]

    return total_loss


def _has_nan_or_inf(grads):
    """Return True if any gradient tensor contains NaN or Inf values.

    This mimics PyTorch's ``GradScaler`` overflow detection: when
    mixed-precision (or ill-conditioned batches) produce non-finite
    gradients the optimiser step should be skipped entirely.

    Args:
        grads (list): Gradient tensors.

    Returns:
        bool
    """
    for g in grads:
        if g is None:
            continue
        g_np = np.asarray(g)
        if not np.all(np.isfinite(g_np)):
            return True
    return False


def _clip_grad_norm(grads, max_norm):
    """Clip gradients by global norm.

    Computes the global L2 norm across all gradient tensors and scales
    them down when the norm exceeds ``max_norm``.

    Args:
        grads (list): List of gradient tensors.
        max_norm (float): Maximum allowed global norm.

    Returns:
        list: Clipped gradient tensors.
    """
    total_norm_sq = sum(ops.sum(g**2) for g in grads if g is not None)
    total_norm = ops.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = ops.minimum(clip_coef, 1.0)
    return [g * clip_coef if g is not None else g for g in grads]


# ---------------------------------------------------------------------------
# Evaluation with COCO metrics
# ---------------------------------------------------------------------------


def evaluate(
    model,
    criterion,
    postprocess,
    data_iterator,
    coco_gt,
    config=None,
    print_freq=10,
):
    """Run the model in evaluation mode and compute COCO metrics.

    Args:
        model: Trained LWDETR model.
        criterion (SetCriterion): Loss module (for computing val losses).
        postprocess (PostProcess): Converts raw outputs to detections.
        data_iterator (iterable): Yields ``(images, targets)`` batches.
        coco_gt (COCO): Ground-truth COCO object for evaluation.
        config: Training/model config with ``segmentation_head`` and
            ``eval_max_dets`` attributes.
        print_freq (int): Logging interval.

    Returns:
        tuple: ``(stats_dict, coco_evaluator)`` where ``stats_dict``
            contains loss averages and ``coco_eval_bbox`` / optionally
            ``coco_eval_masks`` metric arrays.
    """
    from paz.models.detection.dino_v2_object_detection.utils.coco_eval import (
        CocoEvaluator,
    )

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    segmentation_head = getattr(config, "segmentation_head", False)
    eval_max_dets = getattr(config, "eval_max_dets", 500)

    iou_types = ("bbox",) if not segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(coco_gt, list(iou_types), eval_max_dets)

    for step, (images, targets) in enumerate(
        metric_logger.log_every(data_iterator, print_freq, header)
    ):
        images_t = ops.convert_to_tensor(images, dtype="float32")
        outputs = model(images_t, training=False)

        # Compute losses during evaluation (group_detr=1 in eval mode)
        weight_dict = criterion.weight_dict
        group_detr = 1  # eval mode: single query group
        outputs_for_match = {
            k: v for k, v in outputs.items()
            if k not in ("aux_outputs", "enc_outputs")
        }
        indices_main = criterion.matcher(
            outputs_for_match, targets, group_detr=group_detr
        )
        num_boxes = sum(len(t["labels"]) for t in targets)
        sum_group_losses = getattr(criterion, "sum_group_losses", False)
        if not sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes_f = max(float(num_boxes), 1.0)

        enc_indices = None
        if "enc_outputs" in outputs:
            enc_indices = criterion.matcher(
                outputs["enc_outputs"], targets, group_detr=group_detr,
            )

        total_loss = _compute_loss_with_indices(
            outputs, targets, indices_main, [],
            enc_indices, criterion, weight_dict, num_boxes_f,
        )
        metric_logger.update(loss=float(ops.convert_to_numpy(total_loss)))

        # Post-process predictions and feed to COCO evaluator
        orig_sizes = np.stack(
            [t["orig_size"] for t in targets], axis=0
        ).astype("float32")
        target_sizes = ops.convert_to_tensor(orig_sizes, dtype="float32")
        post_result = postprocess(outputs, target_sizes)

        if len(post_result) == 4:
            scores, labels, boxes, masks_list = post_result
        else:
            scores, labels, boxes = post_result
            masks_list = None

        scores_np = ops.convert_to_numpy(scores)
        labels_np = ops.convert_to_numpy(labels)
        boxes_np = ops.convert_to_numpy(boxes)

        res = {}
        for i, t in enumerate(targets):
            image_id = int(t["image_id"].flat[0])
            res[image_id] = {
                "scores": scores_np[i],
                "labels": labels_np[i],
                "boxes": boxes_np[i],
            }
            if masks_list is not None:
                res[image_id]["masks"] = ops.convert_to_numpy(
                    masks_list[i]
                )

        coco_evaluator.update(res)

    print("Averaged stats:", metric_logger)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
    stats["results_json"] = results_json
    stats["coco_eval_bbox"] = (
        coco_evaluator.coco_eval["bbox"].stats.tolist()
    )

    if "segm" in iou_types:
        results_json_segm = coco_extended_metrics(
            coco_evaluator.coco_eval["segm"]
        )
        stats["results_json_segm"] = results_json_segm
        stats["coco_eval_masks"] = (
            coco_evaluator.coco_eval["segm"].stats.tolist()
        )

    return stats, coco_evaluator


# ---------------------------------------------------------------------------
# Metric helpers (no COCO dependency)
# ---------------------------------------------------------------------------


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
    """Sweep confidence thresholds and compute precision, recall, and F1.

    For each threshold, computes per-class metrics and macro-averages
    across classes that have ground-truth annotations.

    Args:
        per_class_data (list[dict]): Per-class scores, matches, ignore flags,
            and total ground-truth counts.
        conf_thresholds (array-like): Confidence thresholds to evaluate.
        classes_with_gt (list[int]): Indices of classes with GT annotations.

    Returns:
        list[dict]: One entry per threshold with ``confidence_threshold``,
            ``macro_f1``, ``macro_precision``, ``macro_recall``, and
            per-class arrays.
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


def coco_extended_metrics(coco_eval):
    """Compute per-class precision/recall by sweeping confidence thresholds.

    Finds the threshold that maximizes macro-F1 across all classes, then
    reports per-class and aggregate metrics at that threshold.

    Args:
        coco_eval (COCOeval): A completed COCOeval object (after
            ``accumulate()`` and ``summarize()``).

    Returns:
        dict: Contains ``class_map`` (per-class stats list), ``map``,
            ``precision``, and ``recall`` at the best threshold.
    """
    iou50_idx = np.argwhere(
        np.isclose(coco_eval.params.iouThrs, 0.50)
    ).item()
    cat_ids = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_idx = 0
    maxdet_idx = 2

    # Unflatten evalImgs into nested dict
    evalImgs_unflat = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        cat_id = e["category_id"]
        area_rng = tuple(e["aRng"])
        img_id = e["image_id"]

        if cat_id not in evalImgs_unflat:
            evalImgs_unflat[cat_id] = {}
        if area_rng not in evalImgs_unflat[cat_id]:
            evalImgs_unflat[cat_id][area_rng] = {}
        evalImgs_unflat[cat_id][area_rng][img_id] = e

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

    per_class_data = []
    for cid in cat_ids:
        dt_scores = []
        dt_matches = []
        dt_ignore = []
        total_gt = 0

        for img_id in coco_eval.params.imgIds:
            e = (
                evalImgs_unflat.get(cid, {})
                .get(area_rng_all, {})
                .get(img_id)
            )
            if e is None:
                continue

            num_dt = len(e["dtIds"])
            gt_ignore = e["gtIgnore"]
            total_gt += sum(1 for ig in gt_ignore if not ig)

            for d in range(num_dt):
                dt_scores.append(e["dtScores"][d])
                dt_matches.append(e["dtMatches"][iou50_idx, d])
                dt_ignore.append(e["dtIgnore"][iou50_idx, d])

        per_class_data.append(
            {
                "scores": np.array(dt_scores),
                "matches": np.array(dt_matches),
                "ignore": np.array(dt_ignore, dtype=bool),
                "total_gt": total_gt,
            }
        )

    conf_thresholds = np.linspace(0.0, 1.0, 101)
    classes_with_gt = [
        k for k in range(num_classes) if per_class_data[k]["total_gt"] > 0
    ]

    confidence_sweep = sweep_confidence_thresholds(
        per_class_data, conf_thresholds, classes_with_gt
    )

    best = max(confidence_sweep, key=lambda x: x["macro_f1"])

    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    per_class = []
    cat_id_to_name = {
        c["id"]: c["name"]
        for c in coco_eval.cocoGt.loadCats(cat_ids)
    }
    for k, cid in enumerate(cat_ids):
        p_slice = coco_eval.eval["precision"][:, :, k, area_idx, maxdet_idx]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)
        ap_per_iou = np.nanmean(p_masked, axis=1)
        ap_50_95 = float(np.nanmean(ap_per_iou))
        ap_50 = float(np.nanmean(p_masked[iou50_idx]))

        if (
            np.isnan(ap_50_95)
            or np.isnan(ap_50)
            or np.isnan(best["per_class_prec"][k])
            or np.isnan(best["per_class_rec"][k])
        ):
            continue

        per_class.append(
            {
                "class": cat_id_to_name.get(int(cid), str(cid)),
                "map@50:95": ap_50_95,
                "map@50": ap_50,
                "precision": best["per_class_prec"][k],
                "recall": best["per_class_rec"][k],
            }
        )

    per_class.append(
        {
            "class": "all",
            "map@50:95": map_50_95,
            "map@50": map_50,
            "precision": best["macro_precision"],
            "recall": best["macro_recall"],
        }
    )

    return {
        "class_map": per_class,
        "map": map_50,
        "precision": best["macro_precision"],
        "recall": best["macro_recall"],
    }

