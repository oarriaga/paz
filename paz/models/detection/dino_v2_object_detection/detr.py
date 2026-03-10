import json
import os
import datetime
import math
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
            LambdaLRSchedule,
            train_one_epoch,
        )
        from paz.models.detection.dino_v2_object_detection.utils.utils import (
            ModelEma,
            BestMetricHolder,
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

        model = self.model.model  # underlying Keras LWDETR

        # ---- Optimizer ----------------------------------------------------
        optimizer = keras.optimizers.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
            clipnorm=config.clip_max_norm if config.clip_max_norm > 0 else None,
        )

        # ---- EMA ----------------------------------------------------------
        if config.use_ema:
            ema_m = ModelEma(model, decay=config.ema_decay, tau=config.ema_tau)
        else:
            ema_m = None

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best_map_holder = BestMetricHolder(use_ema=config.use_ema)
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

        # ---- Epoch loop ---------------------------------------------------
        start_time = time.time()

        for epoch in range(config.epochs):
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
                )

            # Update exponential moving average after each epoch
            if ema_m is not None:
                ema_m.update(model)

            # Save periodic checkpoints
            if config.output_dir:
                ckpt_path = output_dir / "checkpoint.weights.h5"
                model.save_weights(str(ckpt_path))
                if (epoch + 1) % config.checkpoint_interval == 0:
                    model.save_weights(
                        str(output_dir / f"checkpoint{epoch:04}.weights.h5")
                    )

            # Aggregate and log epoch statistics
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }

            # Update best metrics
            train_loss = train_stats.get("loss", train_stats.get("train_loss", 0.0))
            _isbest = best_map_holder.update(train_loss, epoch, is_ema=False)

            log_stats.update(best_map_holder.summary())

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

        # ---- Post-training: apply EMA and fire callbacks ---------------
        total_time = time.time() - start_time
        print(
            f"Training time {datetime.timedelta(seconds=int(total_time))}"
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

        Returns ``None`` if the annotation file does not exist, letting
        the caller fall back to an alternative pipeline.

        Args:
            config: Training configuration with ``dataset_dir``.
            split (str): ``'train'`` or ``'val'``.
            all_kwargs (dict): Extra keyword arguments (used for ``resolution``).

        Returns:
            _COCODataLoader or None: The data loader, or ``None``.
        """
        if not config.dataset_dir:
            return None
        split_dir = os.path.join(config.dataset_dir, split)
        ann_file = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.isfile(ann_file):
            # For "val" split try "valid" (roboflow convention)
            if split == "val":
                split_dir = os.path.join(config.dataset_dir, "valid")
                ann_file = os.path.join(split_dir, "_annotations.coco.json")
            if not os.path.isfile(ann_file):
                return None
        return _COCODataLoader(
            ann_file=ann_file,
            img_dir=split_dir,
            batch_size=config.batch_size * config.grad_accum_steps,
            resolution=all_kwargs.get("resolution", 560),
        )

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
# Minimal COCO data-loader (backend agnostic, numpy-based)
# ---------------------------------------------------------------------------


class _COCODataLoader:
    """Thin data loader that reads COCO annotation JSON + images.

    Yields ``(images_np, targets)`` tuples where ``images_np`` is
    ``(B, H, W, 3)`` float32 and ``targets`` is a list of dicts with
    ``labels`` and ``boxes`` numpy arrays.
    """

    def __init__(self, ann_file, img_dir, batch_size, resolution):
        with open(ann_file, "r") as f:
            self._coco = json.load(f)
        self._img_dir = img_dir
        self._batch_size = batch_size
        self._resolution = resolution

        # Build mappings
        self._images = {img["id"]: img for img in self._coco["images"]}
        self._img_ids = list(self._images.keys())

        self._anns_by_img = defaultdict(list)
        for ann in self._coco.get("annotations", []):
            self._anns_by_img[ann["image_id"]].append(ann)

    def __len__(self):
        return max(1, math.ceil(len(self._img_ids) / self._batch_size))

    def __iter__(self):
        indices = np.random.permutation(len(self._img_ids))
        for start in range(0, len(indices), self._batch_size):
            batch_idx = indices[start : start + self._batch_size]
            images, targets = [], []
            for idx in batch_idx:
                img_id = self._img_ids[idx]
                img_info = self._images[img_id]
                img_path = os.path.join(self._img_dir, img_info["file_name"])
                img_np = self._load_and_resize(img_path)
                images.append(img_np)

                anns = self._anns_by_img.get(img_id, [])
                boxes, labels = [], []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    # Convert to cxcywh normalised
                    cx = (x + w / 2) / img_info["width"]
                    cy = (y + h / 2) / img_info["height"]
                    nw = w / img_info["width"]
                    nh = h / img_info["height"]
                    boxes.append([cx, cy, nw, nh])
                    labels.append(ann["category_id"])

                targets.append(
                    {
                        "labels": np.array(labels, dtype="int64"),
                        "boxes": np.array(boxes, dtype="float32").reshape(-1, 4),
                    }
                )
            images_np = np.stack(images).astype("float32")
            yield images_np, targets

    def _load_and_resize(self, path):
        """Load an image, resize to ``resolution``, and normalise to [0, 1].

        ImageNet channel means and standard deviations are applied after
        dividing by 255.

        Args:
            path (str): Path to the image file.

        Returns:
            np.ndarray: ``(H, W, 3)`` float32 normalised image.
        """
        from PIL import Image as PILImage

        img = PILImage.open(path).convert("RGB")
        img = img.resize(
            (self._resolution, self._resolution), PILImage.BILINEAR
        )
        arr = np.asarray(img, dtype="float32") / 255.0
        # Normalise with ImageNet channel statistics
        means = np.array([0.485, 0.456, 0.406], dtype="float32")
        stds = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - means) / stds
        return arr


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
