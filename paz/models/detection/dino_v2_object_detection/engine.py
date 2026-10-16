import datetime
import functools
import logging
import math
import random
import time
from collections import namedtuple

import numpy as np
import keras
from keras import ops

import jax

from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    AUXILIARY_KEYS,
    apply_lwdetr,
    apply_lwdetr_stateless,
    get_loss,
    update_drop_path,
    update_dropout,
)
from paz.models.detection.dino_v2_object_detection.utils.misc import (
    MetricLogger,
    SmoothedValue,
)

CONFIDENCE_STEPS = 101
DROP_MODES = ("standard", "early", "late")
GRAD_FUNCTION_ATTRIBUTE = "jitted_grad_function"

SubBatch = namedtuple("SubBatch", "images mask targets")
LossInputs = namedtuple("LossInputs", "targets indices aux_indices enc_indices num_boxes")  # fmt: skip
ClassScore = namedtuple("ClassScore", "precision recall f1")


def build_lr_lambda(num_training_steps_per_epoch, epochs, warmup_epochs, lr_scheduler="step", lr_drop=100, lr_min_factor=0.0):  # fmt: skip
    keys = ("total_steps", "warmup_steps", "lr_scheduler", "lr_drop", "num_training_steps_per_epoch", "lr_min_factor")  # fmt: skip
    values = (num_training_steps_per_epoch * epochs, int(num_training_steps_per_epoch * warmup_epochs), lr_scheduler, lr_drop, num_training_steps_per_epoch, lr_min_factor)  # fmt: skip
    return functools.partial(compute_lr_multiplier, **dict(zip(keys, values)))


def compute_lr_multiplier(current_step, total_steps, warmup_steps, lr_scheduler, lr_drop, num_training_steps_per_epoch, lr_min_factor):  # fmt: skip
    if current_step < warmup_steps:
        multiplier = float(current_step) / float(max(1, warmup_steps))
    elif lr_scheduler == "cosine":
        args = (current_step, warmup_steps, total_steps, lr_min_factor)
        multiplier = compute_cosine_multiplier(*args)
    else:
        drop_step = lr_drop * num_training_steps_per_epoch
        multiplier = 1.0 if current_step < drop_step else 0.1
    return multiplier


def compute_cosine_multiplier(current_step, warmup_steps, total_steps, lr_min_factor):  # fmt: skip
    span = float(max(1, total_steps - warmup_steps))
    progress = float(current_step - warmup_steps) / span
    decay = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_factor + (1 - lr_min_factor) * decay


# Not a Layer/Model subclass: Keras hands the optimizer a schedule object and
# calls it per step, so this owns mutable schedule state a function cannot.
class LambdaLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, lr_lambda):
        super().__init__()
        self.base_lr = base_lr
        self.lr_lambda = lr_lambda

    def __call__(self, step):
        return self.base_lr * self.lr_lambda(int(step))

    def get_config(self):
        return {"base_lr": self.base_lr}


def build_drop_schedule(drop_rate, epochs, num_steps_per_epoch, cutoff_epoch=0, mode="standard", schedule="constant"):  # fmt: skip
    assert mode in DROP_MODES
    total_steps = epochs * num_steps_per_epoch
    early_steps = cutoff_epoch * num_steps_per_epoch
    late_steps = (epochs - cutoff_epoch) * num_steps_per_epoch
    if mode == "standard":
        final_schedule = np.full(total_steps, drop_rate, dtype="float32")
    elif mode == "early":
        head = build_early_drop_head(drop_rate, early_steps, schedule)
        tail = np.full(late_steps, 0, dtype="float32")
        final_schedule = np.concatenate((head, tail))
    else:
        assert schedule in ("constant",)
        head = np.full(early_steps, 0, dtype="float32")
        tail = np.full(late_steps, drop_rate, dtype="float32")
        final_schedule = np.concatenate((head, tail))
    assert len(final_schedule) == total_steps
    return final_schedule


def build_early_drop_head(drop_rate, early_steps, schedule):
    assert schedule in ("constant", "linear")
    if schedule == "constant":
        head = np.full(early_steps, drop_rate, dtype="float32")
    else:
        head = np.linspace(drop_rate, 0, early_steps, dtype="float32")
    return head


def build_epoch_logger():
    metric_logger = MetricLogger(delimiter="  ")
    learning_rate = SmoothedValue(window_size=1, fmt="{value:.6f}")
    metric_logger.add_meter("lr", learning_rate)
    return metric_logger


def train_one_epoch(model, criterion, optimizer, data_iterator, num_steps, epoch, clip_max_norm=0.1, print_freq=10, lr_multipliers=None, ema_m=None, grad_accum_steps=1, multi_scale_config=None, drop_path_schedule=None, dropout_schedule=None, vit_encoder_num_layers=None, use_mixed_precision=False):  # fmt: skip
    metric_logger = build_epoch_logger()
    header = f"Epoch: [{epoch}]"
    schedules = (drop_path_schedule, dropout_schedule, vit_encoder_num_layers)
    scales = multi_scale_config["scales"] if multi_scale_config else None
    step_args = (clip_max_norm, lr_multipliers, optimizer, model, ema_m)
    start_time = time.time()
    batches = metric_logger.log_every(data_iterator, print_freq, header)
    for step, (images, targets) in enumerate(batches):
        apply_epoch_schedules(model, epoch * num_steps + step, *schedules)
        images, mask = unpack_epoch_images(images)
        if scales is not None:
            images, mask = resize_multi_scale_batch(images, mask, scales, step)
        args = (model, images, mask, targets, grad_accum_steps)
        state = accumulate_epoch_gradients(*args, use_mixed_precision, criterion)  # fmt: skip
        run_optimizer_step(metric_logger, state, step, *step_args)
        if step >= num_steps - 1:
            break
    report_epoch_time(header, time.time() - start_time, num_steps)
    return {k: meter.global_avg() for k, meter in metric_logger.meters.items()}


def run_optimizer_step(metric_logger, state, step, clip_max_norm, lr_multipliers, optimizer, model, ema_m):  # fmt: skip
    grads, accumulated_loss, updated_non_trainable = state
    # A single host sync per step; the loss is only needed for logging.
    loss = float(ops.convert_to_numpy(accumulated_loss))
    if has_nan_or_inf(grads):
        warn_gradient_overflow(metric_logger, loss, step)
    else:
        args = (grads, clip_max_norm, lr_multipliers, optimizer, model)
        apply_epoch_gradients(*args, updated_non_trainable)
        update_exponential_moving_average(ema_m, model)
        raise_on_non_finite_loss(loss)
        metric_logger.update(loss=loss, lr=read_current_lr(optimizer))


def warn_gradient_overflow(metric_logger, loss, step):
    # Gradient overflow: skip the optimiser step entirely, matching the
    # GradScaler behaviour of mixed-precision training.
    message = "NaN/Inf gradients at step %d - skipping update"
    logging.getLogger(__name__).warning(message, step)
    metric_logger.update(loss=loss, lr=0.0, grad_overflow=1.0)


def update_exponential_moving_average(ema_m, model):
    if ema_m is not None:
        ema_m.update(model)


def raise_on_non_finite_loss(loss):
    if not math.isfinite(loss):
        raise ValueError(f"Loss is {loss}, stopping training")


def report_epoch_time(header, elapsed, num_steps):
    total = datetime.timedelta(seconds=int(elapsed))
    per_step = elapsed / max(1, num_steps)
    print(f"{header} Total time: {total} ({per_step:.4f} s / it)")


def apply_epoch_schedules(model, global_step, drop_path_schedule, dropout_schedule, vit_encoder_num_layers):  # fmt: skip
    rate = read_schedule_rate(drop_path_schedule, global_step)
    if rate is not None:
        update_drop_path(model, rate, vit_encoder_num_layers)
    rate = read_schedule_rate(dropout_schedule, global_step)
    if rate is not None:
        update_dropout(model, rate)


def read_schedule_rate(schedule, global_step):
    rate = None
    if schedule is not None and global_step < len(schedule):
        rate = float(schedule[global_step])
    return rate


def unpack_epoch_images(images):
    # images may be a plain array or a (tensor, mask) tuple
    mask = None
    if isinstance(images, (list, tuple)) and len(images) == 2:
        images, mask = images
        mask = ops.convert_to_tensor(mask, dtype="bool")
    return ops.convert_to_tensor(images, dtype="float32"), mask


def resize_multi_scale_batch(images, mask, scales, step):
    random.seed(step)
    scale = random.choice(scales)
    images = ops.image.resize(images, (scale, scale))
    if mask is not None:
        mask = resize_mask(mask, scale)
    return images, mask


def resize_mask(mask, scale):
    expanded = ops.cast(mask[:, :, :, None], "float32")
    size = (scale, scale)
    resized = ops.image.resize(expanded, size, interpolation="nearest")
    return ops.cast(resized[:, :, :, 0], "bool")


def slice_sub_batch(images, mask, targets, step, size):
    start = step * size
    stop = start + size
    sub_mask = mask[start:stop] if mask is not None else None
    return SubBatch(images[start:stop], sub_mask, targets[start:stop])


def build_model_input(sub_batch, use_mixed_precision=False):
    images = sub_batch.images
    if use_mixed_precision:
        images = ops.cast(images, "bfloat16")
    if sub_batch.mask is None:
        model_input = images
    else:
        model_input = (images, sub_batch.mask)
    return model_input


def accumulate_epoch_gradients(model, images, mask, targets, grad_accum_steps, use_mixed_precision, criterion):  # fmt: skip
    grad_fn = read_jitted_grad_fn(model, criterion, use_mixed_precision)
    sub_batch_size = int(images.shape[0]) // grad_accum_steps
    scale = 1.0 / grad_accum_steps
    accumulated_grads = None
    # Accumulated on device; converted once by the caller so gradient
    # accumulation does not stall JAX dispatch with a sync per sub-step.
    accumulated_loss = ops.convert_to_tensor(0.0, dtype="float32")
    updated_non_trainable = None
    for step in range(grad_accum_steps):
        sub_batch = slice_sub_batch(images, mask, targets, step, sub_batch_size)
        loss_inputs = match_epoch_targets(model, sub_batch, criterion)
        args = (model, sub_batch, use_mixed_precision, grad_fn)
        grads, loss, updated_non_trainable = compute_sub_batch_gradients(*args, loss_inputs)  # fmt: skip
        grads = [gradient * scale for gradient in grads]
        accumulated_loss = accumulated_loss + loss * scale
        accumulated_grads = add_gradients(accumulated_grads, grads)
    return accumulated_grads, accumulated_loss, updated_non_trainable


def add_gradients(accumulated, grads):
    if accumulated is None:
        total = grads
    else:
        total = [a + g for a, g in zip(accumulated, grads)]
    return total


def match_epoch_targets(model, sub_batch, criterion):
    # Phase 1 - eager forward plus Hungarian matching, both host-side.
    outputs = apply_lwdetr(model, build_model_input(sub_batch), training=True)
    targets = sub_batch.targets
    main = {k: v for k, v in outputs.items() if k not in AUXILIARY_KEYS}
    indices = criterion.matcher(main, targets, group_detr=criterion.group_detr)
    aux_indices = match_aux_indices(outputs, targets, criterion)
    enc_indices = match_encoder_indices(outputs, targets, criterion)
    num_boxes = count_matched_boxes(targets, criterion)
    return LossInputs(targets, indices, aux_indices, enc_indices, num_boxes)


def match_aux_indices(outputs, targets, criterion):
    aux_indices = []
    for aux in outputs.get("aux_outputs", []):
        matched = criterion.matcher(aux, targets, group_detr=criterion.group_detr)  # fmt: skip
        aux_indices.append(matched)
    return aux_indices


def match_encoder_indices(outputs, targets, criterion):
    enc_indices = None
    if "enc_outputs" in outputs:
        encoded = outputs["enc_outputs"]
        enc_indices = criterion.matcher(encoded, targets, group_detr=criterion.group_detr)  # fmt: skip
    return enc_indices


def count_matched_boxes(targets, criterion):
    num_boxes = sum(len(target["labels"]) for target in targets)
    if not getattr(criterion, "sum_group_losses", False):
        num_boxes = num_boxes * criterion.group_detr
    return max(float(num_boxes), 1.0)


def cast_outputs_to_float32(outputs):
    casted = {}
    for key, value in outputs.items():
        typed = hasattr(value, "dtype")
        casted[key] = ops.cast(value, "float32") if typed else value
    return casted


def build_jitted_grad_fn(model, criterion, use_mixed_precision):
    # model and criterion are captured instead of passed: a Keras Model is
    # neither a pytree nor hashable, so jax.jit can accept it as neither a
    # traced nor a static argument. Only array pytrees cross the boundary.
    def compute_loss(trainable_values, non_trainable_values, forward_input, loss_inputs):  # fmt: skip
        args = (model, trainable_values, non_trainable_values, forward_input)
        outputs, updated = apply_lwdetr_stateless(*args, training=True)
        if use_mixed_precision:
            outputs = cast_outputs_to_float32(outputs)
        loss = compute_loss_with_indices(outputs, criterion, loss_inputs)
        return loss, updated

    return jax.jit(jax.value_and_grad(compute_loss, has_aux=True))


def read_jitted_grad_fn(model, criterion, use_mixed_precision):
    # Cached on the model so one trace serves the whole run: rebuilding the
    # transform per step would discard the trace cache every step, which is
    # slower than staying eager.
    key = (id(criterion), use_mixed_precision)
    cached_key, cached = getattr(model, GRAD_FUNCTION_ATTRIBUTE, (None, None))
    if cached_key != key:
        cached = build_jitted_grad_fn(model, criterion, use_mixed_precision)
        setattr(model, GRAD_FUNCTION_ATTRIBUTE, (key, cached))
    return cached


def compute_sub_batch_gradients(model, sub_batch, use_mixed_precision, grad_fn, loss_inputs):  # fmt: skip
    # Phase 2 - traced forward, loss and gradients, all inside one jit.
    trainable_values = [v.value for v in model.trainable_variables]
    non_trainable_values = [v.value for v in model.non_trainable_variables]
    forward_input = build_model_input(sub_batch, use_mixed_precision)
    args = (non_trainable_values, forward_input, loss_inputs)
    values, grads = grad_fn(trainable_values, *args)
    loss, updated_non_trainable = values
    if use_mixed_precision:
        grads = [ops.cast(gradient, "float32") for gradient in grads]
    return grads, loss, updated_non_trainable


def add_weighted(total, losses, weight_dict, suffix):
    for key, value in losses.items():
        weight = weight_dict.get(key + suffix)
        if weight is not None:
            total = total + value * weight
    return total


def add_weighted_losses(total, outputs, targets, indices, num_boxes, criterion, suffix):  # fmt: skip
    for loss_type in criterion.loss_types:
        args = (loss_type, outputs, targets, indices, num_boxes, criterion)
        losses = get_loss(*args)
        total = add_weighted(total, losses, criterion.weight_dict, suffix)
    return total


def read_aux_indices(loss_inputs, index):
    aux_indices = loss_inputs.aux_indices
    if index < len(aux_indices):
        indices = aux_indices[index]
    else:
        indices = loss_inputs.indices
    return indices


def add_aux_losses(total, outputs, loss_inputs, num_boxes, criterion):
    for index, aux in enumerate(outputs.get("aux_outputs", [])):
        indices = read_aux_indices(loss_inputs, index)
        args = (aux, loss_inputs.targets, indices, num_boxes, criterion)
        total = add_weighted_losses(total, *args, f"_{index}")
    return total


def compute_encoder_loss(loss_type, outputs, loss_inputs, num_boxes, criterion):
    kwargs = {"log": False} if loss_type == "labels" else {}
    args = (loss_type, outputs["enc_outputs"], loss_inputs.targets)
    tail = (loss_inputs.enc_indices, num_boxes, criterion)
    return get_loss(*args, *tail, **kwargs)


def add_encoder_losses(total, outputs, loss_inputs, num_boxes, criterion):
    if "enc_outputs" in outputs and loss_inputs.enc_indices is not None:
        for loss_type in criterion.loss_types:
            args = (loss_type, outputs, loss_inputs, num_boxes, criterion)
            losses = compute_encoder_loss(*args)
            total = add_weighted(total, losses, criterion.weight_dict, "_enc")
    return total


def compute_loss_with_indices(outputs, criterion, loss_inputs):
    num_boxes = ops.convert_to_tensor(loss_inputs.num_boxes, dtype="float32")
    total = ops.convert_to_tensor(0.0, dtype="float32")
    args = (outputs, loss_inputs.targets, loss_inputs.indices, num_boxes)
    total = add_weighted_losses(total, *args, criterion, "")
    total = add_aux_losses(total, outputs, loss_inputs, num_boxes, criterion)
    return add_encoder_losses(total, outputs, loss_inputs, num_boxes, criterion)


@jax.jit
def compute_gradients_are_finite(grads):
    finite = [ops.all(ops.isfinite(g)) for g in grads if g is not None]
    return ops.all(ops.stack(finite))


def has_nan_or_inf(grads):
    # One host sync for the whole gradient pytree instead of one per tensor.
    return not bool(ops.convert_to_numpy(compute_gradients_are_finite(grads)))


@jax.jit
def clip_grad_norm(grads, max_norm):
    total_norm = ops.sqrt(sum(ops.sum(g**2) for g in grads if g is not None))
    clip_coefficient = ops.minimum(max_norm / (total_norm + 1e-6), 1.0)
    return [g * clip_coefficient if g is not None else g for g in grads]


def apply_epoch_gradients(grads, clip_max_norm, lr_multipliers, optimizer, model, updated_non_trainable):  # fmt: skip
    if clip_max_norm > 0:
        grads = clip_grad_norm(grads, clip_max_norm)
    if lr_multipliers is not None:
        grads = scale_gradients(grads, lr_multipliers, model)
    optimizer.apply(grads, model.trainable_variables)
    # Sync non-trainable vars (e.g. BatchNorm running stats)
    for variable, value in zip(model.non_trainable_variables, updated_non_trainable):  # fmt: skip
        variable.assign(value)


def scale_gradients(grads, lr_multipliers, model):
    variables = model.trainable_variables
    return [g * lr_multipliers.get(v.path, 1.0) for g, v in zip(grads, variables)]  # fmt: skip


def read_current_lr(optimizer):
    learning_rate = getattr(optimizer, "learning_rate", None)
    if learning_rate is None:
        value = 0.0
    elif callable(learning_rate):
        value = float(learning_rate(optimizer.iterations))
    else:
        value = float(learning_rate)
    return value


def read_iou_types(config):
    if getattr(config, "segmentation_head", False):
        iou_types = ("bbox", "segm")
    else:
        iou_types = ("bbox",)
    return iou_types


def evaluate(model, criterion, postprocess, data_iterator, coco_gt, config=None, print_freq=10):  # fmt: skip
    from paz.models.detection.dino_v2_object_detection.utils.coco_eval import (
        CocoEvaluator,
    )
    metric_logger = MetricLogger(delimiter="  ")
    iou_types = read_iou_types(config)
    max_detections = getattr(config, "eval_max_dets", 500)
    coco_evaluator = CocoEvaluator(coco_gt, list(iou_types), max_detections)
    losses = []
    batches = metric_logger.log_every(data_iterator, print_freq, "Test:")
    for images, targets in batches:
        args = (model, images, targets, criterion)
        outputs, total_loss = evaluate_forward_loss(*args)
        losses.append(total_loss)
        results = postprocess_eval_batch(outputs, targets, postprocess)
        coco_evaluator.update(results)
    record_eval_losses(metric_logger, losses)
    stats = aggregate_eval_stats(metric_logger, coco_evaluator, iou_types)
    return stats, coco_evaluator


def record_eval_losses(metric_logger, losses):
    # One host sync for the whole eval loop instead of one per step.
    if losses:
        for value in ops.convert_to_numpy(ops.stack(losses)):
            metric_logger.update(loss=float(value))


def evaluate_forward_loss(model, images, targets, criterion):
    images = ops.convert_to_tensor(images, dtype="float32")
    outputs = apply_lwdetr(model, images, training=False)
    main = {k: v for k, v in outputs.items() if k not in AUXILIARY_KEYS}
    # Eval mode always uses a single query group.
    indices = criterion.matcher(main, targets, group_detr=1)
    enc_indices = match_evaluation_encoder_indices(outputs, targets, criterion)
    num_boxes = count_evaluation_boxes(targets, criterion)
    loss_inputs = LossInputs(targets, indices, [], enc_indices, num_boxes)
    total_loss = compute_loss_with_indices(outputs, criterion, loss_inputs)
    return outputs, total_loss


def match_evaluation_encoder_indices(outputs, targets, criterion):
    enc_indices = None
    if "enc_outputs" in outputs:
        encoded = outputs["enc_outputs"]
        enc_indices = criterion.matcher(encoded, targets, group_detr=1)
    return enc_indices


def count_evaluation_boxes(targets, criterion):
    num_boxes = sum(len(target["labels"]) for target in targets)
    if not getattr(criterion, "sum_group_losses", False):
        num_boxes = num_boxes * 1
    return max(float(num_boxes), 1.0)


def postprocess_eval_batch(outputs, targets, postprocess):
    sizes = np.stack([t["orig_size"] for t in targets], axis=0)
    target_sizes = ops.convert_to_tensor(sizes.astype("float32"), dtype="float32")  # fmt: skip
    result = postprocess(outputs, target_sizes)
    masks_list = result[3] if len(result) == 4 else None
    scores = ops.convert_to_numpy(result[0])
    labels = ops.convert_to_numpy(result[1])
    boxes = ops.convert_to_numpy(result[2])
    return build_coco_results(targets, scores, labels, boxes, masks_list)


def aggregate_eval_stats(metric_logger, coco_evaluator, iou_types):
    print("Averaged stats:", metric_logger)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {k: m.global_avg() for k, m in metric_logger.meters.items()}
    box_eval = coco_evaluator.coco_eval["bbox"]
    stats["results_json"] = coco_extended_metrics(box_eval)
    stats["coco_eval_bbox"] = box_eval.stats.tolist()
    if "segm" in iou_types:
        mask_eval = coco_evaluator.coco_eval["segm"]
        stats["results_json_segm"] = coco_extended_metrics(mask_eval)
        stats["coco_eval_masks"] = mask_eval.stats.tolist()
    return stats


def build_coco_results(targets, scores, labels, boxes, masks_list):
    results = {}
    for index, target in enumerate(targets):
        image_id = int(target["image_id"].flat[0])
        entry = {"scores": scores[index], "labels": labels[index]}
        entry["boxes"] = boxes[index]
        if masks_list is not None:
            entry["masks"] = ops.convert_to_numpy(masks_list[index])
        results[image_id] = entry
    return results


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def score_class_at_threshold(data, threshold):
    selected = (data["scores"] >= threshold) & ~data["ignore"]
    matches = data["matches"][selected]
    true_positive = np.sum(matches != 0)
    false_positive = np.sum(matches == 0)
    false_negative = data["total_gt"] - true_positive
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    return ClassScore(precision, recall, safe_ratio(2 * precision * recall, precision + recall))  # fmt: skip


def build_macro_scores(precisions, recalls, f1_scores, classes_with_gt):
    macro = (0.0, 0.0, 0.0)
    if classes_with_gt:
        selected = [f1_scores[index] for index in classes_with_gt]
        macro = np.mean(precisions[classes_with_gt]), np.mean(recalls[classes_with_gt]), np.mean(selected)  # fmt: skip
    return macro


def summarize_threshold(threshold, scores, classes_with_gt):
    precisions = np.array([score.precision for score in scores])
    recalls = np.array([score.recall for score in scores])
    f1_scores = [score.f1 for score in scores]
    args = (precisions, recalls, f1_scores, classes_with_gt)
    macro_precision, macro_recall, macro_f1 = build_macro_scores(*args)
    summary = {"confidence_threshold": threshold, "macro_f1": macro_f1}
    summary["macro_precision"] = macro_precision
    summary["macro_recall"] = macro_recall
    summary["per_class_prec"] = precisions
    summary["per_class_rec"] = recalls
    return summary


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):  # fmt: skip
    results = []
    for threshold in conf_thresholds:
        scores = [score_class_at_threshold(d, threshold) for d in per_class_data]  # fmt: skip
        results.append(summarize_threshold(threshold, scores, classes_with_gt))
    return results


def coco_extended_metrics(coco_eval):
    iou50_index = np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.50)).item()
    category_ids = coco_eval.params.catIds
    area_index, maxdet_index = 0, 2
    grouped = group_eval_images(coco_eval)
    args = (coco_eval, grouped, iou50_index, category_ids, area_index)
    per_class_data = collect_per_class_data(*args)
    with_ground_truth = [index for index in range(len(category_ids))
                         if per_class_data[index]["total_gt"] > 0]
    thresholds = np.linspace(0.0, 1.0, CONFIDENCE_STEPS)
    sweep = sweep_confidence_thresholds(per_class_data, thresholds, with_ground_truth)  # fmt: skip
    best = max(sweep, key=lambda entry: entry["macro_f1"])
    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])
    args = (coco_eval, category_ids, best, iou50_index, area_index)
    per_class = build_per_class_metrics(*args, maxdet_index, map_50_95, map_50)
    summary = {"class_map": per_class, "map": map_50}
    summary["precision"] = best["macro_precision"]
    summary["recall"] = best["macro_recall"]
    return summary


def group_eval_images(coco_eval):
    grouped = {}
    for entry in coco_eval.evalImgs:
        if entry is None:
            continue
        area_range = tuple(entry["aRng"])
        by_category = grouped.setdefault(entry["category_id"], {})
        by_category.setdefault(area_range, {})[entry["image_id"]] = entry
    return grouped


def read_grouped_entry(grouped, category_id, area_range, image_id):
    by_area = grouped.get(category_id, {})
    return by_area.get(area_range, {}).get(image_id)


def collect_detection_records(entry, iou50_index):
    scores, matches, ignore = [], [], []
    for detection in range(len(entry["dtIds"])):
        scores.append(entry["dtScores"][detection])
        matches.append(entry["dtMatches"][iou50_index, detection])
        ignore.append(entry["dtIgnore"][iou50_index, detection])
    return scores, matches, ignore


def collect_category_data(coco_eval, grouped, iou50_index, category_id, area_range):  # fmt: skip
    scores, matches, ignore = [], [], []
    total_ground_truth = 0
    for image_id in coco_eval.params.imgIds:
        args = (grouped, category_id, area_range, image_id)
        entry = read_grouped_entry(*args)
        if entry is None:
            continue
        total_ground_truth += sum(1 for flag in entry["gtIgnore"] if not flag)
        records = collect_detection_records(entry, iou50_index)
        scores, matches, ignore = extend_records((scores, matches, ignore), records)  # fmt: skip
    data = {"scores": np.array(scores), "matches": np.array(matches)}
    data["ignore"] = np.array(ignore, dtype=bool)
    data["total_gt"] = total_ground_truth
    return data


def extend_records(collected, records):
    for destination, source in zip(collected, records):
        destination.extend(source)
    return collected


def collect_per_class_data(coco_eval, grouped, iou50_index, category_ids, area_index):  # fmt: skip
    area_range = tuple(coco_eval.params.areaRng[area_index])
    per_class_data = []
    for category_id in category_ids:
        args = (coco_eval, grouped, iou50_index, category_id, area_range)
        per_class_data.append(collect_category_data(*args))
    return per_class_data


def build_class_entry(coco_eval, index, area_index, maxdet_index, iou50_index, best, names, category_id):  # fmt: skip
    precision = coco_eval.eval["precision"]
    sliced = precision[:, :, index, area_index, maxdet_index]
    masked = np.where(sliced > -1, sliced, np.nan)
    average = float(np.nanmean(np.nanmean(masked, axis=1)))
    average_50 = float(np.nanmean(masked[iou50_index]))
    class_precision = best["per_class_prec"][index]
    class_recall = best["per_class_rec"][index]
    values = (average, average_50, class_precision, class_recall)
    entry = None
    if not any(np.isnan(value) for value in values):
        entry = {"class": names.get(int(category_id), str(category_id))}
        entry["map@50:95"] = average
        entry["map@50"] = average_50
        entry["precision"] = class_precision
        entry["recall"] = class_recall
    return entry


def build_all_class_entry(best, map_50_95, map_50):
    entry = {"class": "all", "map@50:95": map_50_95, "map@50": map_50}
    entry["precision"] = best["macro_precision"]
    entry["recall"] = best["macro_recall"]
    return entry


def build_per_class_metrics(coco_eval, category_ids, best, iou50_index, area_index, maxdet_index, map_50_95, map_50):  # fmt: skip
    categories = coco_eval.cocoGt.loadCats(category_ids)
    names = {c["id"]: c["name"] for c in categories}
    per_class = []
    for index, category_id in enumerate(category_ids):
        args = (coco_eval, index, area_index, maxdet_index, iou50_index)
        entry = build_class_entry(*args, best, names, category_id)
        if entry is not None:
            per_class.append(entry)
    per_class.append(build_all_class_entry(best, map_50_95, map_50))
    return per_class
