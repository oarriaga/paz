import functools
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

NO_METRIC_MESSAGE = "Early stopping: No valid mAP metric found in log_stats."
NO_HOOK_MESSAGE = "Model does not have stop_training attribute or request_early_stop method."  # fmt: skip


def EarlyStoppingCallback(model, patience=5, min_delta=0.001, use_ema=False, verbose=True, segmentation_head=False):  # fmt: skip
    ns = SimpleNamespace()
    keys = ("patience", "min_delta", "use_ema", "verbose", "model", "segmentation_head")  # fmt: skip
    values = (patience, min_delta, use_ema, verbose, model, segmentation_head)
    for key, value in zip(keys, values):
        setattr(ns, key, value)
    ns.best_map = 0.0
    ns.counter = 0
    ns.update = functools.partial(update_early_stopping, ns)
    return ns


def read_bbox_map(log_stats, segmentation_head, key):
    value = log_stats.get(key)
    listed = isinstance(value, (list, tuple)) and len(value) > 0
    current_map = None
    if listed and not segmentation_head:
        current_map = value[0]
    elif not listed and isinstance(value, (float, int)):
        current_map = value
    return current_map


def read_mask_map(log_stats, key):
    value = log_stats.get(key)
    current_map = None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        current_map = value[0]
    elif isinstance(value, (float, int)):
        current_map = value
    return current_map


def extract_map(log_stats, segmentation_head, prefix):
    current_map = read_bbox_map(log_stats, segmentation_head, prefix + "test_coco_eval_bbox")  # fmt: skip
    mask_map = None
    if segmentation_head:
        mask_map = read_mask_map(log_stats, prefix + "test_coco_eval_masks")
    return current_map if mask_map is None else mask_map


def select_current_map(ns, regular_map, ema_map):
    both = regular_map is not None and ema_map is not None
    if both and ns.use_ema:
        selected = ema_map, "EMA"
    elif both:
        selected = max(regular_map, ema_map), "max(regular, EMA)"
    elif ema_map is not None:
        selected = ema_map, "EMA"
    else:
        selected = regular_map, "regular"
    return selected


def print_early_stopping_status(ns, current_map, metric_source):
    difference = current_map - ns.best_map
    head = f"Early stopping: Current mAP ({metric_source}): "
    body = f"{current_map:.4f}, Best: {ns.best_map:.4f}, "
    print(head + body + f"Diff: {difference:.4f}, Min delta: {ns.min_delta}")


def update_early_stopping(ns, log_stats):
    regular_map = extract_map(log_stats, ns.segmentation_head, "")
    ema_map = extract_map(log_stats, ns.segmentation_head, "ema_")
    current_map, metric_source = select_current_map(ns, regular_map, ema_map)
    if current_map is None and ns.verbose:
        print(NO_METRIC_MESSAGE)
    if current_map is not None:
        if ns.verbose:
            print_early_stopping_status(ns, current_map, metric_source)
        apply_early_stopping_decision(ns, current_map, metric_source)


def record_improvement(ns, current_map, metric_source):
    ns.best_map = current_map
    ns.counter = 0
    message = f"Early stopping: mAP improved to {current_map:.4f} "
    logger.info(message + f"using {metric_source} metric")


def record_stagnation(ns, current_map):
    ns.counter += 1
    if ns.verbose:
        head = "Early stopping: No improvement in mAP for "
        counts = f"{ns.counter} epochs "
        best = f"(best: {ns.best_map:.4f}, "
        print(head + counts + best + f"current: {current_map:.4f})")


def request_model_stop(model):
    if model and hasattr(model, "stop_training"):
        model.stop_training = True
    elif model and hasattr(model, "request_early_stop"):
        model.request_early_stop()
    else:
        logger.warning(NO_HOOK_MESSAGE)


def apply_early_stopping_decision(ns, current_map, metric_source):
    if current_map > ns.best_map + ns.min_delta:
        record_improvement(ns, current_map, metric_source)
    else:
        record_stagnation(ns, current_map)
    if ns.counter >= ns.patience:
        head = "Early stopping triggered: No improvement above "
        print(head + f"{ns.min_delta} threshold for {ns.patience} epochs")
        request_model_stop(ns.model)
