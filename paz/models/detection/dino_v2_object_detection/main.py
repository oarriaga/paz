import os
import re
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

# ---- weights directory ---------------------------------------------

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
    """Instantiate a backbone (``Joiner``) from a ``ModelConfig``.

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
    """Instantiate a ``Transformer`` decoder from a ``ModelConfig``.

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
    """Build the ``HungarianMatcher`` layer.

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
    """Build a complete ``LWDETR`` model from a ``ModelConfig``.

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


# ---- Per-variable weight-decay exclusion --------------------------------

_WD_EXEMPT_KEYWORDS = ("gamma", "pos_embed", "rel_pos", "bias",
                       "norm", "embeddings")


def get_backbone_no_weight_decay_vars(model):
    """Return backbone variables that should be excluded from weight decay.
    ``get_dinov2_weight_decay_rate`` sets
    ``weight_decay=0`` for backbone parameters whose names contain any of
    ``gamma``, ``pos_embed``, ``rel_pos``, ``bias``, ``norm``, or
    ``embeddings``.  Decoder and head parameters always use the default
    weight decay.

    This function identifies the corresponding variables so they
    can be passed to ``optimizer.exclude_from_weight_decay(var_list=...)``.

    Args:
        model: Built LWDETR model.

    Returns:
        list: ``Variable`` objects to exclude from weight decay.
    """
    excluded = []
    for var in model.trainable_variables:
        path = var.path
        # Only backbone encoder variables are eligible for exemption
        if "joiner/backbone/encoder/" not in path:
            continue
        if "projector" in path:
            continue
        if any(kw in path for kw in _WD_EXEMPT_KEYWORDS):
            excluded.append(var)
    return excluded


# ---- Differential learning-rate multipliers ------------------------------

_LAYER_PATTERN = re.compile(r"/layer_(\d+)/")


def get_param_lr_multipliers(model, train_config, model_config=None):
    """Compute per-variable gradient multipliers for differential LR.

    The base learning rate is specified in ``train_config.lr``.  To
    achieve per-group learning rates,
    we pre-multiply each gradient by a variable-specific multiplier so that
    ``base_lr × multiplier == effective_lr``.

    Groups (matching ``rfdetr/util/get_param_dicts.py``):

    * **Backbone encoder** — variables under ``joiner/backbone/encoder/``
      (excluding the projector).  Per-layer decay is applied via
      ``lr_vit_layer_decay`` and the group is also scaled by
      ``lr_component_decay²``.
    * **Decoder** — variables under ``transformer/transformer_decoder/``.
      Scaled by ``lr_component_decay``.
    * **Everything else** — heads, projector, other transformer params.
      Multiplier = 1.0.

    Parameters
    ----------
    model : LWDETR
        The built LWDETR model.
    train_config : TrainConfig or object
        Must have ``lr``, ``lr_encoder``, ``lr_vit_layer_decay``, and
        ``lr_component_decay`` attributes.  If *model_config* is ``None``,
        must also have ``out_feature_indexes``.
    model_config : ModelConfig or None, optional
        When provided, ``out_feature_indexes`` is read from here instead
        of *train_config*.
    """
    lr = train_config.lr
    lr_encoder = train_config.lr_encoder
    lr_vit_layer_decay = train_config.lr_vit_layer_decay
    lr_component_decay = train_config.lr_component_decay

    cfg_for_indexes = model_config if model_config is not None else train_config

    num_layers = cfg_for_indexes.out_feature_indexes[-1] + 2

    multipliers = {}

    for var in model.trainable_variables:
        path = var.path

        # ---- backbone encoder variables --------------------------------
        if "joiner/backbone/encoder/" in path and "projector" not in path:
            if "embeddings" in path:
                layer_id = 0
            else:
                m = _LAYER_PATTERN.search(path)
                if m:
                    layer_id = int(m.group(1)) + 1
                else:
                    # Final layernorm or other non-block params inside
                    # the encoder — gives these decay=1.0
                    layer_id = num_layers + 1

            decay = lr_vit_layer_decay ** (num_layers + 1 - layer_id)
            multiplier = (lr_encoder / lr) * decay * (lr_component_decay ** 2)

        # ---- decoder variables -----------------------------------------
        elif "transformer/transformer_decoder/" in path:
            multiplier = lr_component_decay

        # ---- everything else (heads, projector, etc.) ------------------
        else:
            multiplier = 1.0

        multipliers[path] = multiplier

    return multipliers


# ---- Model wrapper -------------------------------------------------------


class Model:
    """High-level wrapper around the LWDETR model.

    * Builds the ``LWDETR`` from a ``ModelConfig``.
    * Optionally loads pre-trained weights (via the weight-porting utilities).
    * Exposes ``predict`` for numpy-in / numpy-out inference.

    Attributes:
        config (ModelConfig): Architecture configuration.
        resolution (int): Input image resolution.
        model (LWDETR): Underlying model.
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

        # Auto-load pre-ported weights (.weights.h5 or .keras)
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
        post_result = self.postprocess(
            outputs,
            kops.convert_to_tensor(
                np.array([[images.shape[1], images.shape[2]]] * images.shape[0]),
                dtype="float32",
            ),
        )

        # Handle both detection-only (3 returns) and segmentation (4 returns)
        if len(post_result) == 4:
            scores, labels, boxes, masks_list = post_result
        else:
            scores, labels, boxes = post_result
            masks_list = None

        # Convert to numpy and filter by confidence threshold
        scores = ops.convert_to_numpy(scores)
        labels = ops.convert_to_numpy(labels)
        boxes = ops.convert_to_numpy(boxes)

        results = []
        for i in range(images.shape[0]):
            keep = scores[i] > threshold
            result = {
                "boxes": boxes[i][keep],
                "scores": scores[i][keep],
                "labels": labels[i][keep],
            }
            if masks_list is not None:
                masks_i = ops.convert_to_numpy(masks_list[i])
                result["masks"] = masks_i[keep]
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Detection-head re-initialisation
    # ------------------------------------------------------------------

    def reinitialize_detection_head(self, num_classes):
        """Rebuild the model with a new number of classes, preserving
        all backbone and transformer weights.


        Args:
            num_classes (int): New number of object categories.
        """
        import tempfile
        from dataclasses import replace as _dc_replace

        old_num_classes = self.config.num_classes

        # Step 2a — Save current weights to a temp file
        # Also snapshot values positionally for verification
        old_weights = [w.numpy().copy() for w in self.model.weights]
        old_shapes = [tuple(w.shape) for w in self.model.weights]
        old_count = len(old_weights)

        # Snapshot class_embed weights for tiling
        old_class_weights = {}
        for w in self.model.weights:
            if "class_embed" in w.path:
                old_class_weights[w.path] = w.numpy().copy()

        tmpdir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmpdir, "reinit_checkpoint.weights.h5")
        self.model.save_weights(tmp_path)

        # Step 2b — Rebuild model with new num_classes
        updated_config = _dc_replace(self.config, num_classes=num_classes)
        self.config = updated_config
        self.model = build_model_from_config(updated_config)
        self.postprocess = PostProcess(num_select=updated_config.num_select)

        # Step 2c — Materialise all layers via dummy forward pass
        res = updated_config.resolution
        dummy = np.ones((1, res, res, 3), dtype="float32") * 0.5
        self.model(dummy, training=False)

        # Step 2d — Restore weights via load_weights with skip_mismatch
        self.model.load_weights(tmp_path, skip_mismatch=True)

        # Clean up temp file
        try:
            os.remove(tmp_path)
            os.rmdir(tmpdir)
        except OSError:
            pass

        # Step 2e — Tile old class_embed weights into new heads
        # Instead of leaving mismatched heads randomly initialised,
        # tile (repeat) the pretrained weights along the class axis.
        for w in self.model.weights:
            if w.path in old_class_weights:
                old_val = old_class_weights[w.path]
                new_shape = tuple(w.shape)
                if old_val.shape == new_shape:
                    continue  # shapes match, already restored
                # Tile along the last axis (classes dimension)
                # Kernel: (hidden, old_C) → (hidden, new_C)
                # Bias:   (old_C,) → (new_C,)
                if old_val.ndim == 2:
                    reps = int(np.ceil(new_shape[1] / old_val.shape[1]))
                    tiled = np.tile(old_val, (1, reps))[:, :new_shape[1]]
                elif old_val.ndim == 1:
                    reps = int(np.ceil(new_shape[0] / old_val.shape[0]))
                    tiled = np.tile(old_val, reps)[:new_shape[0]]
                else:
                    continue
                w.assign(tiled)
                logger.debug(
                    "Tiled class_embed weight %s: %s → %s",
                    w.path, old_val.shape, new_shape,
                )

        # Step 2f — Validate restoration
        new_weights = [w.numpy() for w in self.model.weights]
        new_shapes = [tuple(w.shape) for w in self.model.weights]

        if len(new_weights) != old_count:
            logger.warning(
                "reinitialize_detection_head: weight count changed "
                "from %d to %d", old_count, len(new_weights),
            )

        # Count restored vs changed by comparing positionally
        restored = 0
        shape_changed = 0
        for i in range(min(old_count, len(new_weights))):
            if old_shapes[i] == new_shapes[i]:
                max_diff = float(np.max(np.abs(old_weights[i] - new_weights[i])))
                if max_diff < 1e-5:
                    restored += 1
            else:
                shape_changed += 1

        # Guard: at least old_count - 30 should be restored
        # (group_detr=13 → 14 Dense layers × 2 = 28 class_embed vars)
        min_expected = old_count - 30
        if restored < min_expected:
            raise RuntimeError(
                f"Too few weights restored: {restored} < {min_expected}. "
                f"shape_changed={shape_changed}. "
                f"Possible architecture mismatch between builds."
            )

        # If num_classes actually changed, at least one var must differ
        if num_classes != old_num_classes and shape_changed == 0:
            logger.warning(
                "reinitialize_detection_head: num_classes changed but "
                "no weight shapes differed — head may not have been "
                "reinitialised."
            )

        logger.info(
            "reinitialize_detection_head: restored=%d, "
            "shape_changed=%d (of %d total)",
            restored, shape_changed, len(new_weights),
        )
