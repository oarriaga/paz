import json
import os
import datetime
import shutil
import time
import functools
from collections import defaultdict, namedtuple
from logging import getLogger
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import keras
from keras import ops

from paz.models.detection.dino_v2_object_detection.config import (
    ModelConfig,
    TrainConfig,
    SegmentationTrainConfig,
    RFDETRBaseConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRLargeConfig,
    RFDETRXLargeConfig,
    RFDETR2XLargeConfig,
    RFDETRSegPreviewConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegMediumConfig,
    RFDETRSegLargeConfig,
    RFDETRSegXLargeConfig,
    RFDETRSeg2XLargeConfig,
)
from paz.models.detection.dino_v2_object_detection.main import (
    Model,
    build_criterion_from_config,
    get_backbone_no_weight_decay_vars,
    get_param_lr_multipliers,
)
from paz.models.detection.dino_v2_object_detection.utils.coco_classes import (
    COCO_CLASSES,
)
from paz.models.detection.dino_v2_object_detection.utils.metrics import (
    MetricsPlotSink,
    MetricsTensorBoardSink,
    MetricsWandBSink,
)

logger = getLogger(__name__)

MEANS = np.array([0.485, 0.456, 0.406], dtype="float32")
STDS = np.array([0.229, 0.224, 0.225], dtype="float32")

COCO_NUM_CLASSES = 90
MIN_TRAIN_BATCHES = 5
COCO_JSON_FILES = ("coco_json", "roboflow")

RESUME_MESSAGE = "Loaded training state: epoch=%d, best_map_5095=%.4f, best_map_ema_5095=%.4f"  # fmt: skip
BAD_STATE_MESSAGE = "Failed to load training_state.json: %s. Starting from epoch 0."  # fmt: skip
MISSING_STATE_MESSAGE = "resume=True but training_state.json not found in %s. Starting from epoch 0."  # fmt: skip
LOADED_CHECKPOINT_MESSAGE = "Loaded model weights from checkpoint. Resuming from epoch %d"  # fmt: skip
MISSING_CHECKPOINT_MESSAGE = "training_state.json found but checkpoint.weights.h5 missing. Starting from epoch 0."  # fmt: skip
MISSING_OPTIMIZER_MESSAGE = "optimizer_state.npz not found. Optimizer starts fresh."  # fmt: skip
SMALL_DATASET_MESSAGE = "Training with uniform sampler because dataset is too small: %d < %d"  # fmt: skip
SPAWN_WARNING = "Setting num_workers to 0 because the script is not wrapped in `if __name__ == '__main__':`. This is required for multiprocessing with the 'spawn' start method."  # fmt: skip
INVALID_DATASET_MESSAGE = "Invalid dataset_file: {!r}. Use 'coco_json' for COCO-format annotations or 'coco' for the standard 80-class COCO dataset."  # fmt: skip

TrainingSetup = namedtuple("TrainingSetup", "model criterion postprocess optimizer lr_multipliers ema_m output_dir best_map_holder start_epoch data_loader_train data_loader_val num_training_steps multi_scale_config drop_path_schedule dropout_schedule vit_encoder_num_layers")  # fmt: skip
DatasetConfig = namedtuple("DatasetConfig", "dataset_file dataset_dir square_resize_div_64 multi_scale expanded_scales do_random_resize_via_padding patch_size num_windows segmentation_head")  # fmt: skip


def RFDETR(model_config_factory=ModelConfig, train_config_factory=TrainConfig, size=None, **kwargs):  # fmt: skip
    ns = SimpleNamespace()
    ns.means, ns.stds, ns.size = MEANS, STDS, size
    ns.model_config_factory = model_config_factory
    ns.train_config_factory = train_config_factory
    ns.model_config = model_config_factory(**kwargs)
    ns.model = Model(ns.model_config)
    ns.resolution = ns.model_config.resolution
    ns.callbacks = defaultdict(list)
    ns.stop_early = False
    keys = ("get_model_config", "get_train_config", "class_names", "predict", "request_early_stop", "train_from_config", "train")  # fmt: skip
    values = (get_model_config, get_train_config, resolve_class_names, predict_detections, request_early_stop, train_from_config, train_model)  # fmt: skip
    for key, value in zip(keys, values):
        setattr(ns, key, functools.partial(value, ns))
    return ns


def get_model_config(ns, **kwargs):
    return ns.model_config_factory(**kwargs)


def get_train_config(ns, **kwargs):
    return ns.train_config_factory(**kwargs)


def resolve_class_names(ns):
    names = COCO_CLASSES
    model_names = getattr(ns.model, "class_names", None)
    if model_names:
        names = {index + 1: name for index, name in enumerate(model_names)}
    return names


def predict_detections(ns, images, threshold=0.5):
    if isinstance(images, list):
        images = np.stack(images)
    if images.ndim == 3:
        images = images[np.newaxis]
    if images.dtype == np.uint8:
        images = images.astype("float32") / 255.0
    return ns.model.predict(images, threshold=threshold)


def request_early_stop(ns):
    ns.stop_early = True
    print("Early stopping requested, will complete current epoch and stop")


def train_model(ns, **kwargs):
    config = get_train_config(ns, **kwargs)
    train_from_config(ns, config, **kwargs)


def train_from_config(ns, config, **kwargs):
    setup = prepare_training(ns, config, kwargs)
    coco_gt = read_coco_ground_truth(setup.data_loader_val)
    start_time = time.time()
    for epoch in range(setup.start_epoch, config.epochs):
        run_training_epoch(ns, epoch, config, setup, coco_gt)
        if ns.stop_early:
            print(f"Early stopping at epoch {epoch}")
            break
    finalize_training(ns, config, setup, start_time)


def prepare_training(ns, config, kwargs):
    from paz.models.detection.dino_v2_object_detection.utils.utils import (
        BestMetricHolder,
    )
    num_classes, class_names = resolve_num_classes(ns, config)
    args = (ns, config, class_names, num_classes, kwargs)
    all_kwargs, train_dict = merge_train_configs(*args)
    register_metric_sinks(ns, config, train_dict)
    criterion, postprocess = build_criterion_from_config(ns.model_config, config)  # fmt: skip
    model = ns.model.model
    apply_backbone_lora(model, config)
    multipliers = get_param_lr_multipliers(model, config, model_config=ns.model_config)  # fmt: skip
    ema_m = build_ema(model, config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    holder = BestMetricHolder(0.0, "large", config.use_ema)
    start_epoch, resume_state = resume_from_checkpoint(config, output_dir, model, ema_m)  # fmt: skip
    data_train, data_val, num_steps = prepare_data_loaders(config, all_kwargs)
    optimizer = build_optimizer(config, model, num_steps)
    restore_optimizer_state(output_dir, model, optimizer, resume_state, start_epoch)  # fmt: skip
    head = (model, criterion, postprocess, optimizer, multipliers, ema_m)
    middle = (output_dir, holder, start_epoch, data_train, data_val, num_steps)
    tail = (build_multi_scale_config(ns, config),)
    return TrainingSetup(*head, *middle, *tail, *build_drop_schedules(config, model, num_steps))  # fmt: skip


def resolve_num_classes(ns, config):
    if config.dataset_file in COCO_JSON_FILES:
        num_classes, class_names = read_annotation_classes(config)
        ns.model.class_names = class_names
    elif config.dataset_file == "coco":
        num_classes, class_names = COCO_NUM_CLASSES, COCO_CLASSES
    else:
        raise ValueError(INVALID_DATASET_MESSAGE.format(config.dataset_file))
    if ns.model_config.num_classes != num_classes:
        ns.model.reinitialize_detection_head(num_classes)
        # Sync model_config so criterion / postprocess see the right count
        ns.model_config = ns.model.config
    return num_classes, class_names


def read_annotation_classes(config):
    path = os.path.join(config.dataset_dir, "train", "_annotations.coco.json")
    with open(path, "r") as handle:
        annotations = json.load(handle)
    categories = annotations["categories"]
    named = [c["name"] for c in categories if c.get("supercategory", "") != "none"]  # fmt: skip
    return len(categories), named


def merge_train_configs(ns, config, class_names, num_classes, kwargs):
    train_dict = config._asdict()
    model_dict = ns.model_config._asdict()
    model_dict.pop("num_classes", None)
    model_dict.pop("class_names", None)
    if train_dict.get("class_names") is None:
        train_dict["class_names"] = class_names
    for key in list(train_dict.keys()):
        model_dict.pop(key, None)
        kwargs.pop(key, None)
    all_kwargs = {**model_dict, **train_dict, **kwargs}
    all_kwargs["num_classes"] = num_classes
    return all_kwargs, train_dict


def register_metric_sinks(ns, config, train_dict):
    plot_sink = MetricsPlotSink(output_dir=config.output_dir)
    ns.callbacks["on_fit_epoch_end"].append(plot_sink.update)
    ns.callbacks["on_train_end"].append(plot_sink.save)
    if config.tensorboard:
        board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
        ns.callbacks["on_fit_epoch_end"].append(board_sink.update)
        ns.callbacks["on_train_end"].append(board_sink.close)
    if config.wandb:
        register_wandb_sink(ns, config, train_dict)
    if config.early_stopping:
        register_early_stopping(ns, config)


def register_wandb_sink(ns, config, train_dict):
    keys = ("output_dir", "project", "run", "config")
    values = (config.output_dir, config.project, config.run, train_dict)
    sink = MetricsWandBSink(**dict(zip(keys, values)))
    ns.callbacks["on_fit_epoch_end"].append(sink.update)
    ns.callbacks["on_train_end"].append(sink.close)


def register_early_stopping(ns, config):
    from paz.models.detection.dino_v2_object_detection.utils.early_stopping import (  # fmt: skip
        EarlyStoppingCallback,
    )
    keys = ("model", "patience", "min_delta", "use_ema", "segmentation_head")
    values = (ns, config.early_stopping_patience, config.early_stopping_min_delta, config.early_stopping_use_ema, config.segmentation_head)  # fmt: skip
    callback = EarlyStoppingCallback(**dict(zip(keys, values)))
    ns.callbacks["on_fit_epoch_end"].append(callback.update)


def apply_backbone_lora(model, config):
    if getattr(config, "backbone_lora", False):
        from paz.models.detection.dino_v2_object_detection.utils.lora import (
            apply_lora_to_backbone,
        )
        keys = ("rank", "lora_alpha", "use_dora")
        values = (getattr(config, "lora_rank", 16), getattr(config, "lora_alpha", 16), getattr(config, "use_dora", True))  # fmt: skip
        apply_lora_to_backbone(model, **dict(zip(keys, values)))
        message = "Applied LoRA (rank=%d, alpha=%d, dora=%s) to backbone."
        logger.info(message, config.lora_rank, config.lora_alpha, config.use_dora)  # fmt: skip


def build_ema(model, config):
    from paz.models.detection.dino_v2_object_detection.utils.utils import (
        ModelEma,
    )
    ema_m = None
    if config.use_ema:
        ema_m = ModelEma(model, decay=config.ema_decay, tau=config.ema_tau)
    return ema_m


def resume_from_checkpoint(config, output_dir, model, ema_m):
    start_epoch, resume_state = 0, None
    if getattr(config, "resume", False):
        start_epoch, resume_state = read_training_state(output_dir)
        args = (output_dir, model, ema_m)
        start_epoch = restore_checkpoint_weights(*args, start_epoch)
    return start_epoch, resume_state


def read_training_state(output_dir):
    path = output_dir / "training_state.json"
    start_epoch, state = 0, None
    if not path.exists():
        logger.warning(MISSING_STATE_MESSAGE, output_dir)
    else:
        try:
            state = json.loads(path.read_text())
            start_epoch = state.get("epoch", 0) + 1
            best = float(state.get("best_map_5095", 0.0))
            best_ema = float(state.get("best_map_ema_5095", 0.0))
            logger.info(RESUME_MESSAGE, start_epoch - 1, best, best_ema)
        except (json.JSONDecodeError, ValueError, KeyError) as error:
            logger.warning(BAD_STATE_MESSAGE, error)
            start_epoch, state = 0, None
    return start_epoch, state


def restore_checkpoint_weights(output_dir, model, ema_m, start_epoch):
    checkpoint_path = output_dir / "checkpoint.weights.h5"
    if start_epoch > 0 and checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))
        logger.info(LOADED_CHECKPOINT_MESSAGE, start_epoch)
        restore_ema_weights(output_dir, model, ema_m)
    elif start_epoch > 0:
        logger.warning(MISSING_CHECKPOINT_MESSAGE)
        start_epoch = 0
    return start_epoch


def restore_ema_weights(output_dir, model, ema_m):
    path = output_dir / "ema_weights.npz"
    if ema_m is not None and path.exists():
        stored = np.load(str(path), allow_pickle=True)
        for key in stored.files:
            if key in ema_m.model_weights:
                ema_m.model_weights[key] = stored[key]
        logger.info("Restored EMA weights from checkpoint.")
    elif ema_m is not None:
        # Fall back to seeding the EMA from the freshly loaded weights.
        ema_m.set(model)
        logger.warning("EMA weights not found. Using current model weights.")


def build_optimizer(config, model, num_training_steps):
    from paz.models.detection.dino_v2_object_detection.engine import (
        build_lr_lambda,
        LambdaLRSchedule,
    )
    keys = ("num_training_steps_per_epoch", "epochs", "warmup_epochs", "lr_scheduler", "lr_drop", "lr_min_factor")  # fmt: skip
    values = (num_training_steps, config.epochs, config.warmup_epochs, config.lr_scheduler, config.lr_drop, config.lr_min_factor)  # fmt: skip
    lr_schedule = LambdaLRSchedule(config.lr, build_lr_lambda(**dict(zip(keys, values))))  # fmt: skip
    kwargs = dict(learning_rate=lr_schedule, weight_decay=config.weight_decay)
    optimizer = keras.optimizers.AdamW(**kwargs)
    # Exclude backbone bias/norm/embedding variables from weight decay.
    no_decay_variables = get_backbone_no_weight_decay_vars(model)
    if no_decay_variables:
        optimizer.exclude_from_weight_decay(var_list=no_decay_variables)
    return optimizer


def restore_optimizer_state(output_dir, model, optimizer, resume_state, start_epoch):  # fmt: skip
    path = output_dir / "optimizer_state.npz"
    resumable = resume_state is not None and start_epoch > 0
    if resumable and not path.exists():
        logger.warning(MISSING_OPTIMIZER_MESSAGE)
    elif resumable:
        prime_optimizer_variables(model, optimizer, output_dir)
        assign_optimizer_variables(optimizer, path)
        saved_iterations = resume_state.get("optimizer_iterations", None)
        if saved_iterations is not None:
            optimizer.iterations.assign(int(saved_iterations))
        logger.info("Restored optimizer state (iterations=%s).", saved_iterations)  # fmt: skip


def assign_optimizer_variables(optimizer, path):
    stored = np.load(str(path), allow_pickle=True)
    for variable in optimizer.variables:
        if variable.path in stored.files:
            variable.assign(stored[variable.path])


def prime_optimizer_variables(model, optimizer, output_dir):
    # A dummy step materialises the optimizer slots; it also perturbs the
    # weights, so the checkpoint is reloaded straight after.
    zeros = [ops.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply(zeros, model.trainable_variables)
    checkpoint_path = output_dir / "checkpoint.weights.h5"
    if checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))


def build_multi_scale_config(ns, config):
    from paz.models.detection.dino_v2_object_detection.datasets import (
        compute_multi_scale_scales,
    )
    multi_scale_config = None
    if config.multi_scale and not config.do_random_resize_via_padding:
        model_config = ns.model_config
        args = (model_config.resolution, config.expanded_scales)
        sizes = (model_config.patch_size, model_config.num_windows)
        multi_scale_config = {"scales": compute_multi_scale_scales(*args, *sizes)}  # fmt: skip
    return multi_scale_config


def build_drop_schedules(config, model, num_training_steps):
    from paz.models.detection.dino_v2_object_detection.engine import (
        build_drop_schedule,
    )
    drop_path_schedule = None
    vit_encoder_num_layers = None
    if getattr(config, "drop_path", 0.0) > 0:
        args = (config.drop_path, config.epochs, num_training_steps)
        drop_path_schedule = build_drop_schedule(*args)
        backbone = model.backbone.get_layer("backbone")
        vit_encoder_num_layers = backbone.get_layer("encoder").num_hidden_layers
    dropout_schedule = None
    if getattr(config, "dropout", 0.0) > 0:
        args = (config.dropout, config.epochs, num_training_steps)
        dropout_schedule = build_drop_schedule(*args)
    return drop_path_schedule, dropout_schedule, vit_encoder_num_layers


def prepare_data_loaders(config, all_kwargs):
    # Users may provide a custom data pipeline; otherwise COCO-format
    # datasets are built automatically from ``dataset_dir``.
    data_loader_train = all_kwargs.pop("data_loader_train", None)
    data_loader_val = all_kwargs.pop("data_loader_val", None)
    if data_loader_train is None:
        data_loader_train = build_data_loader(config, "train", all_kwargs)
    if data_loader_val is None:
        data_loader_val = build_data_loader(config, "val", all_kwargs)
    num_training_steps = 1
    if data_loader_train is not None:
        num_training_steps = len(data_loader_train)
    return data_loader_train, data_loader_val, num_training_steps


def read_coco_ground_truth(data_loader_val):
    coco_gt = None
    if data_loader_val is not None:
        dataset = data_loader_val.dataset
        coco_gt = getattr(dataset, "coco", None)
    return coco_gt


def run_training_epoch(ns, epoch, config, setup, coco_gt):
    epoch_start = time.time()
    print(f"\nEpoch [{epoch}/{config.epochs}]")
    train_stats = run_epoch_pass(config, setup, epoch)
    save_epoch_checkpoints(config, setup, epoch)
    log_stats = {f"train_{k}": v for k, v in train_stats.items()}
    log_stats["epoch"] = epoch
    evaluate_and_track(config, setup, coco_gt, epoch, log_stats)
    evaluate_ema(config, setup, coco_gt, epoch, log_stats)
    log_stats.update(setup.best_map_holder.summary())
    save_training_state(config, setup, epoch)
    elapsed = datetime.timedelta(seconds=int(time.time() - epoch_start))
    log_stats["epoch_time"] = str(elapsed)
    write_epoch_log(config, setup.output_dir, log_stats)
    for callback in ns.callbacks["on_fit_epoch_end"]:
        callback(log_stats)


def write_epoch_log(config, output_dir, log_stats):
    if config.output_dir:
        with (output_dir / "log.txt").open("a") as handle:
            handle.write(json.dumps(log_stats) + "\n")


def run_epoch_pass(config, setup, epoch):
    from paz.models.detection.dino_v2_object_detection.engine import (
        train_one_epoch,
    )
    train_stats = {"train_loss": 0.0}
    if setup.data_loader_train is not None:
        keys = ("model", "criterion", "optimizer", "data_iterator", "num_steps", "epoch", "clip_max_norm", "lr_multipliers", "ema_m", "grad_accum_steps", "multi_scale_config", "drop_path_schedule", "dropout_schedule", "vit_encoder_num_layers", "use_mixed_precision")  # fmt: skip
        values = (setup.model, setup.criterion, setup.optimizer, setup.data_loader_train, setup.num_training_steps, epoch, config.clip_max_norm, setup.lr_multipliers, setup.ema_m, config.grad_accum_steps, setup.multi_scale_config, setup.drop_path_schedule, setup.dropout_schedule, setup.vit_encoder_num_layers, getattr(config, "amp", False))  # fmt: skip
        train_stats = train_one_epoch(**dict(zip(keys, values)))
    return train_stats


def save_epoch_checkpoints(config, setup, epoch):
    if config.output_dir:
        output_dir = setup.output_dir
        setup.model.save_weights(str(output_dir / "checkpoint.weights.h5"))
        state = {v.path: v.numpy() for v in setup.optimizer.variables}
        np.savez(str(output_dir / "optimizer_state.npz"), **state)
        if setup.ema_m is not None:
            path = str(output_dir / "ema_weights.npz")
            np.savez(path, **setup.ema_m.model_weights)
        if (epoch + 1) % config.checkpoint_interval == 0:
            name = f"checkpoint{epoch:04}.weights.h5"
            setup.model.save_weights(str(output_dir / name))


def run_validation(config, setup, coco_gt, prefix, log_stats):
    from paz.models.detection.dino_v2_object_detection.engine import (
        evaluate as evaluate_model,
    )
    stats = {}
    if setup.data_loader_val is not None and coco_gt is not None:
        args = (setup.model, setup.criterion, setup.postprocess)
        stats, _ = evaluate_model(*args, setup.data_loader_val, coco_gt, config=config)  # fmt: skip
        log_stats.update({f"{prefix}{k}": v for k, v in stats.items()})
    return stats


def read_map_metric(config, stats):
    key = "coco_eval_masks" if config.segmentation_head else "coco_eval_bbox"
    return stats.get(key, [0.0])[0]


def track_best_checkpoint(config, setup, value, epoch, is_ema):
    improved = setup.best_map_holder.update(value, epoch, is_ema=is_ema)
    name = "checkpoint_best_ema" if is_ema else "checkpoint_best_regular"
    if improved and config.output_dir:
        path = setup.output_dir / f"{name}.weights.h5"
        setup.model.save_weights(str(path))


def evaluate_and_track(config, setup, coco_gt, epoch, log_stats):
    stats = run_validation(config, setup, coco_gt, "test_", log_stats)
    value = read_map_metric(config, stats)
    track_best_checkpoint(config, setup, value, epoch, False)


def evaluate_ema(config, setup, coco_gt, epoch, log_stats):
    if setup.ema_m is not None and config.use_ema:
        model = setup.model
        original = {w.path: w.numpy().copy() for w in model.weights}
        setup.ema_m.apply_to(model)
        stats = run_validation(config, setup, coco_gt, "ema_test_", log_stats)
        value = read_map_metric(config, stats)
        track_best_checkpoint(config, setup, value, epoch, True)
        restore_model_weights(model, original)


def restore_model_weights(model, original):
    for weight in model.weights:
        if weight.path in original:
            weight.assign(original[weight.path])


def read_best_regular(holder, config):
    best = holder.best_regular if config.use_ema else holder.best_all
    return best.best_res


def read_best_ema(holder, config):
    return holder.best_ema.best_res if config.use_ema else 0.0


def save_training_state(config, setup, epoch):
    if config.output_dir:
        holder = setup.best_map_holder
        iterations = int(ops.convert_to_numpy(setup.optimizer.iterations))
        state = {"epoch": epoch, "optimizer_iterations": iterations}
        state["best_map_5095"] = float(read_best_regular(holder, config))
        state["best_map_ema_5095"] = float(read_best_ema(holder, config))
        with (setup.output_dir / "training_state.json").open("w") as handle:
            json.dump(state, handle, indent=2)


def finalize_training(ns, config, setup, start_time):
    elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"Training time {elapsed}")
    merge_backbone_lora(config, setup.model, setup.output_dir)
    if config.output_dir:
        args = (config, setup.output_dir, setup.ema_m, setup.best_map_holder)
        copy_best_total_checkpoint(*args)
    if setup.ema_m is not None:
        setup.ema_m.apply_to(setup.model)
    for callback in ns.callbacks["on_train_end"]:
        callback()


def merge_backbone_lora(config, model, output_dir):
    if getattr(config, "backbone_lora", False):
        from paz.models.detection.dino_v2_object_detection.utils.lora import (
            merge_lora_weights,
        )
        merge_lora_weights(model)
        logger.info("Merged LoRA weights into base model.")
        if config.output_dir:
            path = output_dir / "checkpoint_merged.weights.h5"
            model.save_weights(str(path))
            logger.info("Saved merged checkpoint to %s", path)


def copy_best_total_checkpoint(config, output_dir, ema_m, best_map_holder):
    regular = output_dir / "checkpoint_best_regular.weights.h5"
    source = regular
    if config.use_ema and ema_m is not None:
        source = select_best_checkpoint(output_dir, best_map_holder, regular)
    if source.exists():
        shutil.copy2(str(source), str(output_dir / "checkpoint_best_total.weights.h5"))  # fmt: skip


def select_best_checkpoint(output_dir, best_map_holder, regular):
    ema_path = output_dir / "checkpoint_best_ema.weights.h5"
    regular_value = best_map_holder.best_regular.best_res
    ema_value = best_map_holder.best_ema.best_res
    prefer_ema = best_map_holder.best_ema.isbetter(ema_value, regular_value)
    if prefer_ema and ema_path.exists():
        source = ema_path
    else:
        source = regular
    return source


def build_data_loader(config, split, all_kwargs):
    from paz.models.detection.dino_v2_object_detection.datasets import (
        COCOBatchLoader,
    )
    dataset, ready = build_loader_dataset(config, split, all_kwargs)
    loader = None
    if ready:
        replacement, num_samples = compute_loader_sampling(dataset, config, split)  # fmt: skip
        keys = ("batch_size", "shuffle", "drop_last", "replacement", "num_samples")  # fmt: skip
        values = (config.batch_size * config.grad_accum_steps, split == "train", split == "train", replacement, num_samples)  # fmt: skip
        loader = COCOBatchLoader(dataset, **dict(zip(keys, values)))
        loader = wrap_loader_prefetch(loader, config)
    return loader


def build_loader_dataset(config, split, all_kwargs):
    from paz.models.detection.dino_v2_object_detection.datasets import (
        build_dataset,
    )
    args = resolve_dataset_config(config, split, all_kwargs)
    dataset, dataset_ready = None, False
    if config.dataset_dir:
        try:
            resolution = all_kwargs.get("resolution", 560)
            dataset = build_dataset(split, args, resolution)
            dataset_ready = True
        except (AssertionError, FileNotFoundError):
            dataset_ready = False
    return dataset, dataset_ready


def resolve_dataset_config(config, split, all_kwargs):
    keys = ("dataset_file", "dataset_dir", "square_resize_div_64", "multi_scale", "expanded_scales", "do_random_resize_via_padding", "patch_size", "num_windows", "segmentation_head")  # fmt: skip
    values = (config.dataset_file, config.dataset_dir, config.square_resize_div_64, config.multi_scale if split == "train" else False, config.expanded_scales, config.do_random_resize_via_padding, all_kwargs.get("patch_size", 14), all_kwargs.get("num_windows", 4), getattr(config, "segmentation_head", False))  # fmt: skip
    return DatasetConfig(**dict(zip(keys, values)))


def compute_loader_sampling(dataset, config, split):
    effective_batch_size = config.batch_size * config.grad_accum_steps
    minimum = effective_batch_size * MIN_TRAIN_BATCHES
    replacement, num_samples = False, None
    # Oversample small training sets so an epoch still has enough batches.
    if split == "train" and len(dataset) < minimum:
        logger.info(SMALL_DATASET_MESSAGE, len(dataset), minimum)
        replacement, num_samples = True, minimum
    return replacement, num_samples


def wrap_loader_prefetch(loader, config):
    num_workers = resolve_num_workers(config)
    if num_workers > 0:
        from paz.models.detection.dino_v2_object_detection.datasets.coco import (  # fmt: skip
            PrefetchBatchLoader,
        )
        loader = PrefetchBatchLoader(loader, num_workers=num_workers)
    return loader


def resolve_num_workers(config):
    import multiprocessing
    num_workers = getattr(config, "num_workers", 0)
    spawning = multiprocessing.get_start_method(allow_none=True) == "spawn"
    if num_workers > 0 and spawning and not is_spawn_safe_main():
        num_workers = 0
    return num_workers


def is_spawn_safe_main():
    import warnings
    try:
        import __main__
        named = __main__.__name__ == "__main__"
        safe = hasattr(__main__, "__file__") and named
        if not safe:
            warnings.warn(SPAWN_WARNING, RuntimeWarning)
    except Exception:
        safe = False
    return safe


RFDETR.build_data_loader = build_data_loader


def annotate_variant(builder, size, model_config_factory, train_config_factory):  # fmt: skip
    # Variant metadata stays reachable without building a model, mirroring
    # the class attributes the upstream rfdetr package exposes.
    builder.size = size
    builder.model_config_factory = model_config_factory
    builder.train_config_factory = train_config_factory
    return builder


def RFDETRBase(**kwargs):
    return RFDETR(RFDETRBaseConfig, size="rfdetr-base", **kwargs)


annotate_variant(RFDETRBase, "rfdetr-base", RFDETRBaseConfig, TrainConfig)


def RFDETRNano(**kwargs):
    return RFDETR(RFDETRNanoConfig, size="rfdetr-nano", **kwargs)


annotate_variant(RFDETRNano, "rfdetr-nano", RFDETRNanoConfig, TrainConfig)  # fmt: skip


def RFDETRSmall(**kwargs):
    return RFDETR(RFDETRSmallConfig, size="rfdetr-small", **kwargs)


annotate_variant(RFDETRSmall, "rfdetr-small", RFDETRSmallConfig, TrainConfig)  # fmt: skip


def RFDETRMedium(**kwargs):
    return RFDETR(RFDETRMediumConfig, size="rfdetr-medium", **kwargs)


annotate_variant(RFDETRMedium, "rfdetr-medium", RFDETRMediumConfig, TrainConfig)  # fmt: skip


def RFDETRLarge(**kwargs):
    return RFDETR(RFDETRLargeConfig, size="rfdetr-large", **kwargs)


annotate_variant(RFDETRLarge, "rfdetr-large", RFDETRLargeConfig, TrainConfig)  # fmt: skip


def RFDETRXLarge(**kwargs):
    return RFDETR(RFDETRXLargeConfig, size="rfdetr-xlarge", **kwargs)


annotate_variant(RFDETRXLarge, "rfdetr-xlarge", RFDETRXLargeConfig, TrainConfig)  # fmt: skip


def RFDETR2XLarge(**kwargs):
    return RFDETR(RFDETR2XLargeConfig, size="rfdetr-2xlarge", **kwargs)


annotate_variant(RFDETR2XLarge, "rfdetr-2xlarge", RFDETR2XLargeConfig, TrainConfig)  # fmt: skip


def RFDETRSegPreview(**kwargs):
    return RFDETR(RFDETRSegPreviewConfig, SegmentationTrainConfig, size="rfdetr-seg-preview", **kwargs)  # fmt: skip


annotate_variant(RFDETRSegPreview, "rfdetr-seg-preview", RFDETRSegPreviewConfig, SegmentationTrainConfig)  # fmt: skip


def RFDETRSegNano(**kwargs):
    return RFDETR(RFDETRSegNanoConfig, SegmentationTrainConfig, size="rfdetr-seg-nano", **kwargs)  # fmt: skip


annotate_variant(RFDETRSegNano, "rfdetr-seg-nano", RFDETRSegNanoConfig, SegmentationTrainConfig)  # fmt: skip


def RFDETRSegSmall(**kwargs):
    return RFDETR(RFDETRSegSmallConfig, SegmentationTrainConfig, size="rfdetr-seg-small", **kwargs)  # fmt: skip


annotate_variant(RFDETRSegSmall, "rfdetr-seg-small", RFDETRSegSmallConfig, SegmentationTrainConfig)  # fmt: skip


def RFDETRSegMedium(**kwargs):
    return RFDETR(RFDETRSegMediumConfig, SegmentationTrainConfig, size="rfdetr-seg-medium", **kwargs)  # fmt: skip


annotate_variant(RFDETRSegMedium, "rfdetr-seg-medium", RFDETRSegMediumConfig, SegmentationTrainConfig)  # fmt: skip


def RFDETRSegLarge(**kwargs):
    return RFDETR(RFDETRSegLargeConfig, SegmentationTrainConfig, size="rfdetr-seg-large", **kwargs)  # fmt: skip


annotate_variant(RFDETRSegLarge, "rfdetr-seg-large", RFDETRSegLargeConfig, SegmentationTrainConfig)  # fmt: skip


def RFDETRSegXLarge(**kwargs):
    return RFDETR(RFDETRSegXLargeConfig, SegmentationTrainConfig, size="rfdetr-seg-xlarge", **kwargs)  # fmt: skip


annotate_variant(RFDETRSegXLarge, "rfdetr-seg-xlarge", RFDETRSegXLargeConfig, SegmentationTrainConfig)  # fmt: skip


def RFDETRSeg2XLarge(**kwargs):
    return RFDETR(RFDETRSeg2XLargeConfig, SegmentationTrainConfig, size="rfdetr-seg-2xlarge", **kwargs)  # fmt: skip


annotate_variant(RFDETRSeg2XLarge, "rfdetr-seg-2xlarge", RFDETRSeg2XLargeConfig, SegmentationTrainConfig)  # fmt: skip


VARIANT_REGISTRY = {
    "RFDETRBase": RFDETRBase,
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge": RFDETRLarge,
    "RFDETRXLarge": RFDETRXLarge,
    "RFDETR2XLarge": RFDETR2XLarge,
    "RFDETRSegPreview": RFDETRSegPreview,
    "RFDETRSegNano": RFDETRSegNano,
    "RFDETRSegSmall": RFDETRSegSmall,
    "RFDETRSegMedium": RFDETRSegMedium,
    "RFDETRSegLarge": RFDETRSegLarge,
    "RFDETRSegXLarge": RFDETRSegXLarge,
    "RFDETRSeg2XLarge": RFDETRSeg2XLarge,
}
