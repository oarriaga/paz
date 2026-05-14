import json
import os
import datetime
import math
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from logging import getLogger
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RFDETR:
    """High-level RF-DETR interface.

    Sub-classes override ``get_model_config`` / ``get_train_config`` to
    return the configuration for a specific model variant.

    Attributes:
        means (np.ndarray): ImageNet channel means for normalisation.
        stds (np.ndarray): ImageNet channel standard deviations.
        size (Optional[str]): Human-readable variant identifier.
    """

    means = np.array([0.485, 0.456, 0.406], dtype="float32")
    stds = np.array([0.229, 0.224, 0.225], dtype="float32")
    size: Optional[str] = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)
        self.stop_early = False

    # ---- overridable hooks -----------------------------------------------

    def get_model_config(self, **kwargs):
        return ModelConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

    def get_model(self, config):
        return Model(config)

    # ---- train -----------------------------------------------------------

    def train(self, **kwargs):
        """Train the RF-DETR model.

        Accepts keyword arguments that are forwarded to
        ``get_train_config``.  For example::

            model.train(dataset_dir="path/to/data", epochs=15, batch_size=16, lr=1e-4)
        """
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)

    def request_early_stop(self):
        """Signal the training loop to stop after the current epoch."""
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train_from_config(self, config, **kwargs):
        """Full training loop driven by a ``TrainConfig``.

        Steps:
        - Reads annotations to determine ``num_classes`` / ``class_names``.
        - Reinitialises the detection head when the class count changes.
        - Sets up criterion, optimizer, LR schedule, and EMA.
        - Runs epoch loop with evaluation and checkpointing.
        - Fires all registered callbacks.
        """
        from paz.models.detection.dino_v2_object_detection.engine import (
            build_lr_lambda,
            build_drop_schedule,
            LambdaLRSchedule,
            train_one_epoch,
            evaluate as evaluate_model,
        )
        from paz.models.detection.dino_v2_object_detection.utils.utils import (
            ModelEma,
            BestMetricHolder,
        )
        from paz.models.detection.dino_v2_object_detection.datasets import (
            compute_multi_scale_scales,
        )

        # ---- Determine num_classes / class_names from annotations -----
        if config.dataset_file in ("coco_json", "roboflow"):
            # Generic COCO-format JSON (also covers Roboflow exports)
            ann_path = os.path.join(
                config.dataset_dir, "train", "_annotations.coco.json"
            )
            with open(ann_path, "r") as f:
                anns = json.load(f)
            num_classes = len(anns["categories"])
            class_names = [
                c["name"]
                for c in anns["categories"]
                if c.get("supercategory", "") != "none"
            ]
            self.model.class_names = class_names
        elif config.dataset_file == "coco":
            class_names = COCO_CLASSES
            num_classes = 90
        else:
            raise ValueError(
                f"Invalid dataset_file: {config.dataset_file!r}. "
                f"Use 'coco_json' for COCO-format annotations or 'coco' "
                f"for the standard 80-class COCO dataset."
            )

        if self.model_config.num_classes != num_classes:
            self.model.reinitialize_detection_head(num_classes)
            # Sync model_config so criterion / postprocess see the right count
            self.model_config = self.model.config

        # ---- Merge model and training config dicts --------------------
        train_dict = asdict(config)
        model_dict = asdict(self.model_config)
        model_dict.pop("num_classes", None)
        model_dict.pop("class_names", None)

        if train_dict.get("class_names") is None:
            train_dict["class_names"] = class_names

        for k in list(train_dict.keys()):
            model_dict.pop(k, None)
            kwargs.pop(k, None)

        all_kwargs = {**model_dict, **train_dict, **kwargs, "num_classes": num_classes}

        # ---- Set up metrics logging sinks -----------------------------
        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            tb_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(tb_sink.update)
            self.callbacks["on_train_end"].append(tb_sink.close)

        if config.wandb:
            wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=train_dict,
            )
            self.callbacks["on_fit_epoch_end"].append(wandb_sink.update)
            self.callbacks["on_train_end"].append(wandb_sink.close)

        if config.early_stopping:
            from paz.models.detection.dino_v2_object_detection.utils.early_stopping import (
                EarlyStoppingCallback,
            )

            es_cb = EarlyStoppingCallback(
                model=self,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema,
                segmentation_head=config.segmentation_head,
            )
            self.callbacks["on_fit_epoch_end"].append(es_cb.update)

        # ---- Build loss criterion and post-processing -----------------
        criterion, postprocess = build_criterion_from_config(
            self.model_config,
            config,
        )

        model = self.model.model  # underlying LWDETR

        # ---- Apply LoRA to backbone (before optimizer, after weights) -----
        if getattr(config, "backbone_lora", False):
            from paz.models.detection.dino_v2_object_detection.utils.lora import (
                apply_lora_to_backbone,
            )
            apply_lora_to_backbone(
                model,
                rank=getattr(config, "lora_rank", 16),
                lora_alpha=getattr(config, "lora_alpha", 16),
                use_dora=getattr(config, "use_dora", True),
            )
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d, dora=%s) to backbone.",
                config.lora_rank, config.lora_alpha, config.use_dora,
            )

        # ---- Optimizer (schedule built below after data loading) ----------
        optimizer = None  # placeholder; created after num_training_steps known

        # ---- Differential learning-rate multipliers -----------------------
        lr_multipliers = get_param_lr_multipliers(
            model, config, model_config=self.model_config
        )

        # ---- EMA ----------------------------------------------------------
        if config.use_ema:
            ema_m = ModelEma(model, decay=config.ema_decay, tau=config.ema_tau)
        else:
            ema_m = None

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best_map_holder = BestMetricHolder(
            init_res=0.0, use_ema=config.use_ema, better='large',
        )
        best_map_5095 = 0.0
        best_map_ema_5095 = 0.0
        start_epoch = 0

        # ---- Resume from checkpoint -------------------------------------
        _resume_state = None
        if getattr(config, "resume", False):
            training_state_path = output_dir / "training_state.json"
            checkpoint_path = output_dir / "checkpoint.weights.h5"

            if training_state_path.exists():
                try:
                    with open(training_state_path) as f:
                        state = json.load(f)
                    start_epoch = state.get("epoch", 0) + 1
                    best_map_5095 = float(state.get("best_map_5095", 0.0))
                    best_map_ema_5095 = float(
                        state.get("best_map_ema_5095", 0.0)
                    )
                    _resume_state = state
                    logger.info(
                        "Loaded training state: epoch=%d, "
                        "best_map_5095=%.4f, best_map_ema_5095=%.4f",
                        start_epoch - 1,
                        best_map_5095,
                        best_map_ema_5095,
                    )
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning(
                        "Failed to load training_state.json: %s. "
                        "Starting from epoch 0.",
                        e,
                    )
                    start_epoch = 0
            else:
                logger.warning(
                    "resume=True but training_state.json not found in %s. "
                    "Starting from epoch 0.",
                    output_dir,
                )

            if checkpoint_path.exists() and start_epoch > 0:
                model.load_weights(str(checkpoint_path))
                logger.info(
                    "Loaded model weights from checkpoint. "
                    "Resuming from epoch %d",
                    start_epoch,
                )

                # Restore EMA weights
                ema_weights_path = output_dir / "ema_weights.npz"
                if ema_m is not None and ema_weights_path.exists():
                    ema_data = np.load(
                        str(ema_weights_path), allow_pickle=True
                    )
                    for key in ema_data.files:
                        if key in ema_m.model_weights:
                            ema_m.model_weights[key] = ema_data[key]
                    logger.info("Restored EMA weights from checkpoint.")
                elif ema_m is not None:
                    # Fallback: create fresh EMA from current model
                    ema_m.set(model)
                    logger.warning(
                        "EMA weights not found. Using current model weights."
                    )

            elif start_epoch > 0 and not checkpoint_path.exists():
                logger.warning(
                    "training_state.json found but checkpoint.weights.h5 "
                    "missing. Starting from epoch 0."
                )
                start_epoch = 0
                best_map_5095 = 0.0
                best_map_ema_5095 = 0.0

        # ---- Data loading ------------------------------------------------
        # Users may provide a custom data pipeline.  When ``dataset_dir``
        # is set, COCO-format datasets are built automatically.
        data_loader_train = all_kwargs.pop("data_loader_train", None)
        data_loader_val = all_kwargs.pop("data_loader_val", None)

        if data_loader_train is None:
            data_loader_train = self._build_data_loader(
                config, "train", all_kwargs
            )
        if data_loader_val is None:
            data_loader_val = self._build_data_loader(
                config, "val", all_kwargs
            )

        effective_batch_size = config.batch_size * config.grad_accum_steps
        if data_loader_train is not None:
            num_training_steps = len(data_loader_train)
        else:
            num_training_steps = 1

        # ---- Build LR schedule & optimizer (needs num_training_steps) -----
        lr_lambda = build_lr_lambda(
            num_training_steps_per_epoch=num_training_steps,
            epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            lr_scheduler=config.lr_scheduler,
            lr_drop=config.lr_drop,
            lr_min_factor=config.lr_min_factor,
        )
        lr_schedule = LambdaLRSchedule(config.lr, lr_lambda)

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
        )

        # ---- Exclude backbone bias/norm/embed from weight decay -----------
        no_wd_vars = get_backbone_no_weight_decay_vars(model)
        if no_wd_vars:
            optimizer.exclude_from_weight_decay(var_list=no_wd_vars)

        # ---- Restore optimizer state on resume ----------------------------
        if _resume_state is not None and start_epoch > 0:
            opt_state_path = output_dir / "optimizer_state.npz"
            if opt_state_path.exists():
                # Run a dummy optimizer step to initialise variables
                dummy_grads = [
                    ops.zeros_like(v) for v in model.trainable_variables
                ]
                optimizer.apply(dummy_grads, model.trainable_variables)
                # Reload model weights (dummy step may have changed them)
                checkpoint_path = output_dir / "checkpoint.weights.h5"
                if checkpoint_path.exists():
                    model.load_weights(str(checkpoint_path))

                # Restore optimizer variables
                opt_data = np.load(str(opt_state_path), allow_pickle=True)
                for v in optimizer.variables:
                    if v.path in opt_data.files:
                        v.assign(opt_data[v.path])

                # Restore iteration count for LR schedule
                saved_iters = _resume_state.get("optimizer_iterations", None)
                if saved_iters is not None:
                    optimizer.iterations.assign(int(saved_iters))

                logger.info(
                    "Restored optimizer state (iterations=%s).",
                    saved_iters,
                )
            else:
                logger.warning(
                    "optimizer_state.npz not found. "
                    "Optimizer starts fresh."
                )

        # ---- Per-batch multi-scale config ---------------------------------
        multi_scale_config = None
        if (
            config.multi_scale
            and not config.do_random_resize_via_padding
        ):
            ms_scales = compute_multi_scale_scales(
                self.model_config.resolution,
                config.expanded_scales,
                self.model_config.patch_size,
                self.model_config.num_windows,
            )
            multi_scale_config = {"scales": ms_scales}

        # ---- Drop path schedule ------------------------------------------
        drop_path_schedule = None
        vit_encoder_num_layers = None
        if getattr(config, "drop_path", 0.0) > 0:
            drop_path_schedule = build_drop_schedule(
                config.drop_path,
                config.epochs,
                num_training_steps,
            )
            vit_encoder_num_layers = (
                model.backbone.backbone.encoder.encoder.encoder.num_hidden_layers
            )

        dropout_schedule = None
        if getattr(config, "dropout", 0.0) > 0:
            dropout_schedule = build_drop_schedule(
                config.dropout,
                config.epochs,
                num_training_steps,
            )

        # ---- Epoch loop ---------------------------------------------------
        start_time = time.time()

        # Get COCO ground truth for validation evaluation
        coco_gt = None
        if data_loader_val is not None:
            dataset_val = data_loader_val.dataset
            if hasattr(dataset_val, "coco"):
                coco_gt = dataset_val.coco

        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.time()
            print(f"\nEpoch [{epoch}/{config.epochs}]")

            # Training epoch forward/backward pass
            train_stats = {"train_loss": 0.0}
            if data_loader_train is not None:
                train_stats = train_one_epoch(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    data_iterator=data_loader_train,
                    num_steps=num_training_steps,
                    epoch=epoch,
                    clip_max_norm=config.clip_max_norm,
                    lr_multipliers=lr_multipliers,
                    ema_m=ema_m,
                    grad_accum_steps=config.grad_accum_steps,
                    multi_scale_config=multi_scale_config,
                    drop_path_schedule=drop_path_schedule,
                    dropout_schedule=dropout_schedule,
                    vit_encoder_num_layers=vit_encoder_num_layers,
                    amp=getattr(config, "amp", False),
                )

            # Save periodic checkpoints
            if config.output_dir:
                ckpt_path = output_dir / "checkpoint.weights.h5"
                model.save_weights(str(ckpt_path))

                # Save optimizer state
                opt_state = {
                    v.path: v.numpy()
                    for v in optimizer.variables
                }
                np.savez(
                    str(output_dir / "optimizer_state.npz"),
                    **opt_state,
                )

                # Save EMA weights
                if ema_m is not None:
                    np.savez(
                        str(output_dir / "ema_weights.npz"),
                        **ema_m.model_weights,
                    )

                if (epoch + 1) % config.checkpoint_interval == 0:
                    model.save_weights(
                        str(output_dir / f"checkpoint{epoch:04}.weights.h5")
                    )

            # Aggregate and log epoch statistics
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }

            # ---- COCO evaluation on validation set ----
            map_regular = 0.0
            if data_loader_val is not None and coco_gt is not None:
                test_stats, coco_evaluator = evaluate_model(
                    model, criterion, postprocess, data_loader_val,
                    coco_gt, config=config,
                )
                log_stats.update(
                    {f"test_{k}": v for k, v in test_stats.items()}
                )

                if not config.segmentation_head:
                    map_regular = test_stats.get(
                        "coco_eval_bbox", [0.0]
                    )[0]
                else:
                    map_regular = test_stats.get(
                        "coco_eval_masks", [0.0]
                    )[0]

            is_best_regular = best_map_holder.update(
                map_regular, epoch, is_ema=False,
            )

            # Save best-regular checkpoint when the metric improves
            if is_best_regular and config.output_dir:
                model.save_weights(
                    str(output_dir / "checkpoint_best_regular.weights.h5")
                )

            # EMA evaluation
            if ema_m is not None and config.use_ema:
                # Apply EMA weights, evaluate, then restore
                original_weights = {
                    w.path: w.numpy().copy() for w in model.weights
                }
                ema_m.apply_to(model)

                map_ema = 0.0
                if data_loader_val is not None and coco_gt is not None:
                    ema_test_stats, _ = evaluate_model(
                        model, criterion, postprocess, data_loader_val,
                        coco_gt, config=config,
                    )
                    log_stats.update(
                        {f"ema_test_{k}": v
                         for k, v in ema_test_stats.items()}
                    )
                    if not config.segmentation_head:
                        map_ema = ema_test_stats.get(
                            "coco_eval_bbox", [0.0]
                        )[0]
                    else:
                        map_ema = ema_test_stats.get(
                            "coco_eval_masks", [0.0]
                        )[0]

                is_best_ema = best_map_holder.update(
                    map_ema, epoch, is_ema=True,
                )
                if is_best_ema and config.output_dir:
                    model.save_weights(
                        str(output_dir / "checkpoint_best_ema.weights.h5")
                    )
                # Restore original (non-EMA) weights
                for w in model.weights:
                    if w.path in original_weights:
                        w.assign(original_weights[w.path])

            log_stats.update(best_map_holder.summary())

            # Save training state for resume
            if config.output_dir:
                training_state = {
                    "epoch": epoch,
                    "optimizer_iterations": int(
                        ops.convert_to_numpy(optimizer.iterations)
                    ),
                    "best_map_5095": float(
                        best_map_holder.best_regular.best_res
                        if config.use_ema
                        else best_map_holder.best_all.best_res
                    ),
                    "best_map_ema_5095": float(
                        best_map_holder.best_ema.best_res
                        if config.use_ema
                        else 0.0
                    ),
                }
                with (output_dir / "training_state.json").open("w") as f:
                    json.dump(training_state, f, indent=2)

            epoch_time = time.time() - epoch_start
            log_stats["epoch_time"] = str(
                datetime.timedelta(seconds=int(epoch_time))
            )

            if config.output_dir:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # Fire epoch-end callbacks
            for cb in self.callbacks["on_fit_epoch_end"]:
                cb(log_stats)

            if self.stop_early:
                print(f"Early stopping at epoch {epoch}")
                break

        # ---- Post-training: merge LoRA, best checkpoint & apply EMA ----
        total_time = time.time() - start_time
        print(
            f"Training time {datetime.timedelta(seconds=int(total_time))}"
        )

        # Merge LoRA weights into the base model
        if getattr(config, "backbone_lora", False):
            from paz.models.detection.dino_v2_object_detection.utils.lora import (
                merge_lora_weights,
            )
            merge_lora_weights(model)
            logger.info("Merged LoRA weights into base model.")
            if config.output_dir:
                merged_path = output_dir / "checkpoint_merged.weights.h5"
                model.save_weights(str(merged_path))
                logger.info("Saved merged checkpoint to %s", merged_path)

        # Determine best-total: copy whichever best checkpoint is better
        if config.output_dir:
            best_regular_path = output_dir / "checkpoint_best_regular.weights.h5"
            best_ema_path = output_dir / "checkpoint_best_ema.weights.h5"
            best_total_path = output_dir / "checkpoint_best_total.weights.h5"

            if config.use_ema and ema_m is not None:
                reg_val = best_map_holder.best_regular.best_res
                ema_val = best_map_holder.best_ema.best_res
                best_is_ema = best_map_holder.best_ema.isbetter(
                    ema_val, reg_val,
                )
                if best_is_ema and best_ema_path.exists():
                    shutil.copy2(str(best_ema_path), str(best_total_path))
                elif best_regular_path.exists():
                    shutil.copy2(
                        str(best_regular_path), str(best_total_path),
                    )
            else:
                if best_regular_path.exists():
                    shutil.copy2(
                        str(best_regular_path), str(best_total_path),
                    )

        # Apply EMA weights to the model if available
        if ema_m is not None:
            ema_m.apply_to(model)

        # Fire on_train_end callbacks
        for cb in self.callbacks["on_train_end"]:
            cb()

    # ---- COCO data-loader factory -------------------------------------

    @staticmethod
    def _build_data_loader(config, split, all_kwargs):
        """Build a COCO-format data loader from ``dataset_dir``.

        For training splits, applies small-dataset oversampling when the
        dataset has fewer samples than ``batch_size * grad_accum_steps *
        min_batches``.  When ``num_workers > 0``, wraps the loader with
        a prefetching thread pool.

        Returns ``None`` if the annotation file does not exist, letting
        the caller fall back to an alternative pipeline.

        Args:
            config: Training configuration with ``dataset_dir``.
            split (str): ``'train'`` or ``'val'``.
            all_kwargs (dict): Extra keyword arguments (used for ``resolution``).

        Returns:
            COCOBatchLoader or PrefetchBatchLoader or None.
        """
        import logging
        import multiprocessing
        import warnings
        from paz.models.detection.dino_v2_object_detection.datasets import (
            build_dataset,
            COCOBatchLoader,
        )
        from paz.models.detection.dino_v2_object_detection.datasets.coco import (
            PrefetchBatchLoader,
        )

        _logger = logging.getLogger(__name__)

        if not config.dataset_dir:
            return None

        # Build a lightweight namespace for the dataset builder
        class _Args:
            dataset_file = config.dataset_file
            dataset_dir = config.dataset_dir
            square_resize_div_64 = config.square_resize_div_64
            multi_scale = config.multi_scale if split == "train" else False
            expanded_scales = config.expanded_scales
            do_random_resize_via_padding = config.do_random_resize_via_padding
            patch_size = all_kwargs.get("patch_size", 14)
            num_windows = all_kwargs.get("num_windows", 4)
            segmentation_head = getattr(config, "segmentation_head", False)

        resolution = all_kwargs.get("resolution", 560)
        try:
            dataset = build_dataset(split, _Args(), resolution)
        except (AssertionError, FileNotFoundError):
            return None

        effective_batch_size = config.batch_size * config.grad_accum_steps
        replacement = False
        num_samples = None

        # Small-dataset oversampling (train split only)
        if split == "train":
            min_batches = 5
            if len(dataset) < effective_batch_size * min_batches:
                _logger.info(
                    "Training with uniform sampler because dataset is too "
                    "small: %d < %d",
                    len(dataset), effective_batch_size * min_batches,
                )
                replacement = True
                num_samples = effective_batch_size * min_batches

        loader = COCOBatchLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            replacement=replacement,
            num_samples=num_samples,
        )

        # Multi-process prefetching
        num_workers = getattr(config, "num_workers", 0)
        if num_workers > 0:
            # Safety check for spawn start method
            start_method = multiprocessing.get_start_method(allow_none=True)
            if start_method == "spawn":
                try:
                    import __main__
                    if (not hasattr(__main__, "__file__")
                            or __main__.__name__ != "__main__"):
                        warnings.warn(
                            "Setting num_workers to 0 because the script is "
                            "not wrapped in `if __name__ == '__main__':`. "
                            "This is required for multiprocessing with the "
                            "'spawn' start method.",
                            RuntimeWarning,
                        )
                        num_workers = 0
                except Exception:
                    num_workers = 0

        if num_workers > 0:
            loader = PrefetchBatchLoader(loader, num_workers=num_workers)

        return loader

    # ---- predict ---------------------------------------------------------

    def predict(self, images, threshold=0.5):
        """Run detection on one or more images.

        Args:
            images (np.ndarray or list[np.ndarray]): HWC uint8 or float
                images.  Lists are stacked into a batch.
            threshold (float): Confidence threshold for filtering.

        Returns:
            list[dict]: Per-image results with ``boxes`` (xyxy),
                ``scores``, ``labels``, and optionally ``masks``.
        """
        # Ensure batch dimension
        if isinstance(images, list):
            images = np.stack(images)
        if images.ndim == 3:
            images = images[np.newaxis]

        # Convert uint8 images to float32 [0, 1]
        if images.dtype == np.uint8:
            images = images.astype("float32") / 255.0

        return self.model.predict(images, threshold=threshold)

    # ---- properties ------------------------------------------------------

    @property
    def class_names(self):
        if hasattr(self.model, "class_names") and self.model.class_names:
            return {i + 1: name for i, name in enumerate(self.model.class_names)}
        return COCO_CLASSES

    @property
    def resolution(self):
        return self.model_config.resolution


# ---------------------------------------------------------------------------
# Detection variants
# ---------------------------------------------------------------------------


class RFDETRBase(RFDETR):
    size = "rfdetr-base"

    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)


class RFDETRNano(RFDETR):
    size = "rfdetr-nano"

    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)


class RFDETRSmall(RFDETR):
    size = "rfdetr-small"

    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)


class RFDETRMedium(RFDETR):
    size = "rfdetr-medium"

    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)


class RFDETRLarge(RFDETR):
    size = "rfdetr-large"

    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)


class RFDETRXLarge(RFDETR):
    size = "rfdetr-xlarge"

    def get_model_config(self, **kwargs):
        return RFDETRXLargeConfig(**kwargs)


class RFDETR2XLarge(RFDETR):
    size = "rfdetr-2xlarge"

    def get_model_config(self, **kwargs):
        return RFDETR2XLargeConfig(**kwargs)


# ---------------------------------------------------------------------------
# Segmentation variants
# ---------------------------------------------------------------------------


class RFDETRSegPreview(RFDETR):
    size = "rfdetr-seg-preview"

    def get_model_config(self, **kwargs):
        return RFDETRSegPreviewConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegNano(RFDETR):
    size = "rfdetr-seg-nano"

    def get_model_config(self, **kwargs):
        return RFDETRSegNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegSmall(RFDETR):
    size = "rfdetr-seg-small"

    def get_model_config(self, **kwargs):
        return RFDETRSegSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegMedium(RFDETR):
    size = "rfdetr-seg-medium"

    def get_model_config(self, **kwargs):
        return RFDETRSegMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegLarge(RFDETR):
    size = "rfdetr-seg-large"

    def get_model_config(self, **kwargs):
        return RFDETRSegLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSegXLarge(RFDETR):
    size = "rfdetr-seg-xlarge"

    def get_model_config(self, **kwargs):
        return RFDETRSegXLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRSeg2XLarge(RFDETR):
    size = "rfdetr-seg-2xlarge"

    def get_model_config(self, **kwargs):
        return RFDETRSeg2XLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

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
