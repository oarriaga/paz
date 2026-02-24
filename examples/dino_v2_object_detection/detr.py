from collections import defaultdict
from logging import getLogger
from typing import Optional

import numpy as np

from examples.dino_v2_object_detection.config import (
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
from examples.dino_v2_object_detection.main import Model
from examples.dino_v2_object_detection.utils.coco_classes import (
    COCO_CLASSES,
)

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RFDETR:
    """High-level RF-DETR interface (Keras 3).

    Sub-classes override ``get_model_config`` / ``get_train_config`` to
    return the right variant.
    """

    means = np.array([0.485, 0.456, 0.406], dtype="float32")
    stds = np.array([0.229, 0.224, 0.225], dtype="float32")
    size: Optional[str] = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

    # ---- overridable hooks -----------------------------------------------

    def get_model_config(self, **kwargs):
        return ModelConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

    def get_model(self, config):
        return Model(config)

    # ---- predict ---------------------------------------------------------

    def predict(self, images, threshold=0.5):
        """Run detection on a batch of images.

        Parameters
        ----------
        images : np.ndarray or list[np.ndarray]
            HWC uint8/float images.  Lists are stacked into a batch.
        threshold : float
            Confidence threshold.

        Returns
        -------
        list[dict] : per-image results with ``boxes``, ``scores``, ``labels``
            (and optionally ``masks``).
        """
        if isinstance(images, list):
            images = np.stack(images)
        if images.ndim == 3:
            images = images[np.newaxis]

        # uint8 → float
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
