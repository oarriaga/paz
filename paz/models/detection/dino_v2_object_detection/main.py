import os
import math
from dataclasses import asdict
from logging import getLogger

import numpy as np
import keras
from keras import ops

from paz.models.detection.dino_v2_object_detection.config import ModelConfig, TrainConfig
from paz.models.detection.dino_v2_object_detection.models.lwdetr.lwdetr import (
    LWDETR,
    SetCriterion,
    PostProcess,
)
from paz.models.detection.dino_v2_object_detection.models.backbone import (
    build_backbone as _build_backbone,
)
from paz.models.detection.dino_v2_object_detection.models.transformer_decoder_head.transformer import (
    Transformer,
)
from paz.models.detection.dino_v2_object_detection.models.segmentation_head.segmentation_head_keras import (
    SegmentationHead,
)
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher
from paz.models.detection.dino_v2_object_detection.utils.utils import (
    ModelEma,
    BestMetricHolder,
)

logger = getLogger(__name__)

# ---- Keras weights directory ---------------------------------------------

# Directory containing pre-ported .weights.h5 files (relative to project root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
KERAS_WEIGHTS_DIR = os.path.join(_PROJECT_ROOT, "lwdetr_keras_weights")


def resolve_weights_path(filename):
    """Return the absolute path for a ``.weights.h5`` / ``.keras`` file.

    Search order:
    1. ``filename`` as-is (absolute or CWD-relative).
    2. Inside ``KERAS_WEIGHTS_DIR``.
    """
    if filename is None:
        return None
    if os.path.isfile(filename):
        return filename
    candidate = os.path.join(KERAS_WEIGHTS_DIR, filename)
    if os.path.isfile(candidate):
        return candidate
    return None


# ---- Model builders ------------------------------------------------------


def build_backbone_from_config(cfg):
    """Instantiate a Keras backbone (``Joiner``) from a ``ModelConfig``.

    Returns a ``Joiner`` that wraps ``Backbone`` + ``PositionEmbeddingSine``.

    Args:
        cfg (ModelConfig): Architecture configuration.

    Returns:
        Joiner: Combined backbone and positional encoding.
    """
    return _build_backbone(
        encoder=cfg.encoder,
        hidden_dim=cfg.hidden_dim,
        out_channels=cfg.hidden_dim,
        out_feature_indexes=cfg.out_feature_indexes,
        projector_scale=cfg.projector_scale,
        layer_norm=cfg.layer_norm,
        target_shape=(cfg.resolution, cfg.resolution),
        gradient_checkpointing=cfg.gradient_checkpointing,
        load_dinov2_weights=cfg.pretrain_weights is None,
        patch_size=cfg.patch_size,
        num_windows=cfg.num_windows,
        positional_encoding_size=cfg.positional_encoding_size,
    )


def build_transformer_from_config(cfg):
    """Instantiate a Keras ``Transformer`` decoder from a ``ModelConfig``.

    Args:
        cfg (ModelConfig): Architecture configuration.

    Returns:
        Transformer: Configured transformer decoder.
    """
    num_feature_levels = len(cfg.projector_scale)
    dim_feedforward = getattr(cfg, "dim_feedforward", 2048)
    return Transformer(
        d_model=cfg.hidden_dim,
        sa_nhead=cfg.sa_nheads,
        ca_nhead=cfg.ca_nheads,
        num_queries=cfg.num_queries,
        num_decoder_layers=cfg.dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
        group_detr=cfg.group_detr,
        two_stage=cfg.two_stage,
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg.dec_n_points,
        lite_refpoint_refine=cfg.lite_refpoint_refine,
        decoder_norm_type="LN",
        bbox_reparam=cfg.bbox_reparam,
    )


def build_segmentation_head_from_config(cfg):
    """Build optional segmentation head, or ``None`` if disabled.

    Args:
        cfg (ModelConfig): Architecture configuration.

    Returns:
        SegmentationHead or None: The segmentation head layer.
    """
    if not cfg.segmentation_head:
        return None
    return SegmentationHead(
        in_dim=cfg.hidden_dim,
        num_blocks=cfg.dec_layers,
        downsample_ratio=cfg.mask_downsample_ratio,
    )


def build_matcher_from_config(cfg, train_cfg=None):
    """Build the ``HungarianMatcher`` Keras layer.

    Args:
        cfg (ModelConfig): Architecture configuration.
        train_cfg (TrainConfig or None): Training configuration (unused
            currently, reserved for future cost overrides).

    Returns:
        HungarianMatcher: The bipartite matching layer.
    """
    cost_class = getattr(cfg, "set_cost_class", 2)
    cost_bbox = getattr(cfg, "set_cost_bbox", 5)
    cost_giou = getattr(cfg, "set_cost_giou", 2)
    focal_alpha = getattr(cfg, "focal_alpha", 0.25)
    return HungarianMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        focal_alpha=focal_alpha,
    )


def build_model_from_config(cfg):
    """Build a complete ``LWDETR`` Keras model from a ``ModelConfig``.

    Returns
    -------
    model : LWDETR
    """
    num_classes = cfg.num_classes + 1  # background class

    backbone = build_backbone_from_config(cfg)
    transformer = build_transformer_from_config(cfg)
    seg_head = build_segmentation_head_from_config(cfg)

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=seg_head,
        num_classes=num_classes,
        num_queries=cfg.num_queries,
        aux_loss=True,
        group_detr=cfg.group_detr,
        two_stage=cfg.two_stage,
        lite_refpoint_refine=cfg.lite_refpoint_refine,
        bbox_reparam=cfg.bbox_reparam,
    )
    return model


def build_criterion_from_config(cfg, train_cfg=None):
    """Build ``SetCriterion`` and ``PostProcess`` from configuration.

    Constructs the loss criterion with auxiliary and two-stage weight
    entries, and a post-processor that applies NMS and selects top-K
    predictions.

    Args:
        cfg (ModelConfig): Architecture configuration.
        train_cfg (TrainConfig or None): Training configuration.

    Returns:
        tuple: ``(criterion, postprocess)``.
    """
    matcher = build_matcher_from_config(cfg, train_cfg)

    bbox_loss_coef = 5.0
    giou_loss_coef = 2.0
    focal_alpha = 0.25

    weight_dict = {
        "loss_ce": cfg.cls_loss_coef,
        "loss_bbox": bbox_loss_coef,
        "loss_giou": giou_loss_coef,
    }
    if cfg.segmentation_head and train_cfg is not None:
        weight_dict["loss_mask_ce"] = getattr(train_cfg, "mask_ce_loss_coef", 5.0)
        weight_dict["loss_mask_dice"] = getattr(train_cfg, "mask_dice_loss_coef", 5.0)

    # Auxiliary losses -- create weight entries for each decoder layer
    # except the last.  Iterate over the *base* keys only (not the
    # growing dict) to avoid cascading duplication.
    base_weight_keys = list(weight_dict.items())
    for i in range(cfg.dec_layers - 1):
        weight_dict.update({k + f"_{i}": v for k, v in base_weight_keys})

    # Two-stage encoder losses: duplicate base weights with "_enc" suffix
    if cfg.two_stage:
        weight_dict.update({k + "_enc": v for k, v in base_weight_keys})

    losses = ["labels", "boxes", "cardinality"]
    if cfg.segmentation_head:
        losses.append("masks")

    criterion = SetCriterion(
        num_classes=cfg.num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=focal_alpha,
        loss_types=losses,
        group_detr=cfg.group_detr,
        ia_bce_loss=cfg.ia_bce_loss,
    )
    postprocess = PostProcess(num_select=cfg.num_select)
    return criterion, postprocess


# ---- Model wrapper -------------------------------------------------------


class Model:
    """High-level wrapper around the LWDETR Keras model.

    * Builds the Keras ``LWDETR`` from a ``ModelConfig``.
    * Optionally loads pre-trained weights (via the weight-porting utilities).
    * Exposes ``predict`` for numpy-in / numpy-out inference.

    Attributes:
        config (ModelConfig): Architecture configuration.
        resolution (int): Input image resolution.
        model (LWDETR): Underlying Keras model.
        postprocess (PostProcess): Post-processing layer.
        class_names (list[str] or None): Class label names.
    """

    def __init__(self, config):
        """Initialise the Model wrapper.

        Args:
            config (ModelConfig): Dataclass with all architecture
                hyperparameters.

        Raises:
            TypeError: If *config* is not a ``ModelConfig``.
        """
        if not isinstance(config, ModelConfig):
            raise TypeError(f"Expected ModelConfig, got {type(config)}")

        self.config = config
        self.resolution = config.resolution
        self.model = build_model_from_config(config)
        self.postprocess = PostProcess(num_select=config.num_select)
        self.class_names = None

        # Auto-load pre-ported Keras weights (.weights.h5 or .keras)
        if config.pretrain_weights is not None:
            wpath = resolve_weights_path(config.pretrain_weights)
            if wpath is not None:
        # Build all layers with a dummy forward pass so weights can be loaded
                res = config.resolution
                dummy = np.ones((1, res, res, 3), dtype="float32") * 0.5
                self.model(dummy, training=False)
                self.load_pretrained_weights(wpath)
            else:
                logger.warning(
                    "Pretrained weights '%s' not found. "
                    "Model initialised with random weights.",
                    config.pretrain_weights,
                )

    # ------------------------------------------------------------------
    # Pretrained weight loading
    # ------------------------------------------------------------------

    def load_pretrained_weights(self, weights_path=None):
        """Load weights from a ``.keras`` or ``.weights.h5`` checkpoint.

        Args:
            weights_path (str or None): Path to the weights file.
                If ``None``, falls back to ``config.pretrain_weights``.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        if weights_path is None:
            weights_path = self.config.pretrain_weights
        if weights_path is None:
            return

        ext = os.path.splitext(weights_path)[-1].lower()
        if ext == ".keras":
            self.model = keras.models.load_model(weights_path)
        elif ext in (".h5", ".hdf5"):
            self.model.load_weights(weights_path, skip_mismatch=True)
        else:
            raise ValueError(
                f"Unsupported weight format '{ext}'. "
                "Use .keras or .weights.h5 (or port from .pth first)."
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, images, threshold=0.5):
        """Run inference on one or more images.

        Args:
            images (np.ndarray): ``(B, H, W, 3)`` float32 normalised to [0, 1].
            threshold (float): Confidence threshold.

        Returns:
            list[dict]: Per-image results with ``boxes`` (xyxy, int),
                ``scores``, ``labels``, and optionally ``masks``.
        """
        # Ensure batch dimension
        if images.ndim == 3:
            images = images[np.newaxis]

        # Apply ImageNet normalisation
        means = np.array([0.485, 0.456, 0.406], dtype="float32")
        stds = np.array([0.229, 0.224, 0.225], dtype="float32")
        images = (images - means) / stds

        # Resize images to the model's expected resolution
        from keras import ops as kops

        imgs = kops.convert_to_tensor(images, dtype="float32")
        imgs = kops.image.resize(
            imgs, (self.resolution, self.resolution), antialias=True
        )

        # Forward pass and post-processing
        outputs = self.model(imgs, training=False)
        scores, labels, boxes = self.postprocess(
            outputs,
            kops.convert_to_tensor(
                np.array([[images.shape[1], images.shape[2]]] * images.shape[0]),
                dtype="float32",
            ),
        )

        # Convert to numpy and filter by confidence threshold
        scores = ops.convert_to_numpy(scores)
        labels = ops.convert_to_numpy(labels)
        boxes = ops.convert_to_numpy(boxes)

        results = []
        for i in range(images.shape[0]):
            keep = scores[i] > threshold
            results.append(
                {
                    "boxes": boxes[i][keep],
                    "scores": scores[i][keep],
                    "labels": labels[i][keep],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Detection-head re-initialisation
    # ------------------------------------------------------------------

    def reinitialize_detection_head(self, num_classes):
        """Rebuild the model with a new number of classes.

        Recreates the entire ``LWDETR`` from scratch with the updated
        ``num_classes``.  Existing pretrained backbone and transformer
        weights are discarded; this is intended for fine-tuning on a
        new dataset where weights will be re-initialised.

        Args:
            num_classes (int): New number of object categories.
        """
        from dataclasses import replace as _dc_replace

        updated_config = _dc_replace(self.config, num_classes=num_classes)
        self.config = updated_config
        self.model = build_model_from_config(updated_config)
        self.postprocess = PostProcess(num_select=updated_config.num_select)
