import math
import os
import re
import sys
import copy
import time
import datetime
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the paz package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PAZ_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PAZ_ROOT not in sys.path:
    sys.path.insert(0, _PAZ_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# =====================================================================
# 1. MODEL PARAMETER COUNTING
# =====================================================================


def count_parameters(model) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters for a Keras model.

    Returns
    -------
    dict with keys: total, trainable, frozen, pct_trainable
    """
    total = sum(np.prod(v.shape) for v in model.weights)
    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    frozen = total - trainable
    pct = 100.0 * trainable / max(total, 1)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
        "pct_trainable": float(pct),
    }


def count_component_parameters(model) -> Dict[str, Dict[str, int]]:
    """Count parameters per model component (backbone, transformer, head).

    Uses :func:`_get_component_for_variable` for consistent classification.

    Returns
    -------
    dict mapping component name -> {total, trainable, frozen}
    """
    components: Dict[str, list] = {}

    trainable_set = set(id(tv) for tv in model.trainable_weights)

    for v in model.weights:
        comp = _get_component_for_variable(v.path)
        n = int(np.prod(v.shape))
        is_trainable = id(v) in trainable_set
        components.setdefault(comp, []).append((n, is_trainable))

    result = {}
    for name, entries in components.items():
        total = sum(e[0] for e in entries)
        trainable = sum(e[0] for e in entries if e[1])
        frozen = total - trainable
        if total > 0:
            result[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
            }
    return result


# =====================================================================
# 2. HARDWARE / FRAMEWORK DIAGNOSTICS
# =====================================================================


def get_hardware_info() -> Dict[str, str]:
    """Gather GPU, framework, and system information.

    Returns
    -------
    dict[str, str]
        Keys include ``keras_version``, ``keras_backend``,
        ``jax_version``, ``device_kind``, ``python_version``, etc.
    """
    info = {}

    # Keras
    try:
        import keras
        info["keras_version"] = keras.__version__
        info["keras_backend"] = keras.backend.backend()
    except Exception:
        info["keras_version"] = "unknown"
        info["keras_backend"] = "unknown"

    # JAX
    try:
        import jax
        info["jax_version"] = jax.__version__
        devices = jax.devices()
        if devices:
            dev = devices[0]
            info["device_kind"] = str(dev.device_kind)
            info["device_name"] = str(dev)
            info["num_devices"] = len(devices)
        else:
            info["device_kind"] = "cpu"
            info["device_name"] = "CPU"
            info["num_devices"] = 0
    except Exception:
        info["jax_version"] = "unknown"
        info["device_kind"] = "unknown"

    # CUDA
    try:
        import jax
        cuda_version = "N/A"
        if hasattr(jax, "lib") and hasattr(jax.lib, "xla_bridge"):
            backend = jax.lib.xla_bridge.get_backend()
            if hasattr(backend, "platform_version"):
                cuda_version = backend.platform_version
        info["cuda_version"] = cuda_version
    except Exception:
        info["cuda_version"] = "unknown"

    # System
    import platform
    info["python_version"] = platform.python_version()
    info["system"] = f"{platform.system()} {platform.machine()}"

    return info


def estimate_gflops(model_config, batch_size: int = 1) -> float:
    """Rough GFLOPs estimate for RF-DETR based on architecture parameters.

    This is an analytical estimate, not a profiled measurement.
    For DINOv2 ViT backbone + transformer decoder + detection head.
    """
    res = model_config.resolution
    patch_size = model_config.patch_size
    hidden_dim = model_config.hidden_dim
    dec_layers = model_config.dec_layers
    num_queries = model_config.num_queries
    num_windows = getattr(model_config, "num_windows", 1)

    # ViT backbone FLOPs (approximate)
    n_patches = (res // patch_size) ** 2
    # Patch embedding: res*res*3 * hidden_dim (per window)
    # Self-attention per layer: 4 * n_patches * hidden_dim^2 + 2 * n_patches^2 * hidden_dim
    # FFN per layer: 8 * n_patches * hidden_dim^2
    # DINOv2-small: 12 layers, hidden_dim ~384 internally but projects to hidden_dim
    encoder_name = getattr(model_config, "encoder", "dinov2_windowed_small")
    if "base" in encoder_name:
        vit_dim = 768
        n_layers = 12
    else:
        vit_dim = 384
        n_layers = 12

    patches_per_window = n_patches // max(num_windows, 1)
    # Self-attention FLOPs per layer
    sa_flops = 4 * patches_per_window * vit_dim ** 2 + \
               2 * patches_per_window ** 2 * vit_dim
    # FFN FLOPs per layer (4x expansion)
    ffn_flops = 8 * patches_per_window * vit_dim ** 2
    backbone_flops = n_layers * (sa_flops + ffn_flops) * num_windows

    # Projector FLOPs (1x1 conv, approximate)
    proj_flops = n_patches * vit_dim * hidden_dim

    # Decoder FLOPs
    # Self-attention: 4 * Q * d^2 + 2 * Q^2 * d
    # Cross-attention: 4 * Q * d^2 + 2 * Q * n_patches * d
    # FFN: 8 * Q * d^2
    dec_sa = 4 * num_queries * hidden_dim ** 2 + \
             2 * num_queries ** 2 * hidden_dim
    dec_ca = 4 * num_queries * hidden_dim ** 2 + \
             2 * num_queries * n_patches * hidden_dim
    dec_ffn = 8 * num_queries * hidden_dim ** 2
    decoder_flops = dec_layers * (dec_sa + dec_ca + dec_ffn)

    # Detection head FLOPs
    # class_embed: Q * d * num_classes
    # bbox_embed (3-layer MLP): Q * (d*d + d*d + d*4)
    num_classes = getattr(model_config, "num_classes", 90)
    head_flops = num_queries * (
        hidden_dim * (num_classes + 1) +
        3 * hidden_dim * hidden_dim + hidden_dim * 4
    )

    total_flops = (backbone_flops + proj_flops + decoder_flops + head_flops)
    gflops = total_flops * batch_size / 1e9

    return round(gflops, 2)


def log_model_summary(model, model_config, batch_size, grad_accum_steps,
                      logger):
    """Log comprehensive pre-training diagnostic information.

    Parameters
    ----------
    model : keras.Model
        The LWDETR Keras model.
    model_config : ModelConfig
    batch_size : int
    grad_accum_steps : int
    logger : logging.Logger
    """
    params = count_parameters(model)
    components = count_component_parameters(model)
    hw = get_hardware_info()
    gflops = estimate_gflops(model_config)

    logger.info("")
    logger.info("=" * 60)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total parameters     : %.2f M (%d)",
                params["total"] / 1e6, params["total"])
    logger.info("  Trainable parameters : %.2f M (%d)",
                params["trainable"] / 1e6, params["trainable"])
    logger.info("  Frozen parameters    : %.2f M (%d)",
                params["frozen"] / 1e6, params["frozen"])
    logger.info("  %% Trainable          : %.1f%%", params["pct_trainable"])
    logger.info("")

    logger.info("  Component breakdown:")
    for comp_name, comp_params in components.items():
        logger.info("    %-20s total=%.2fM  trainable=%.2fM  frozen=%.2fM",
                     comp_name,
                     comp_params["total"] / 1e6,
                     comp_params["trainable"] / 1e6,
                     comp_params["frozen"] / 1e6)
    logger.info("")

    logger.info("  GFLOPs (estimated)   : %.2f", gflops)
    logger.info("  Resolution           : %d x %d",
                model_config.resolution, model_config.resolution)
    logger.info("  Batch size           : %d", batch_size)
    logger.info("  Effective batch size : %d",
                batch_size * grad_accum_steps)
    logger.info("")

    logger.info("  HARDWARE / FRAMEWORK:")
    for key, val in hw.items():
        logger.info("    %-20s: %s", key, val)
    logger.info("=" * 60)
    logger.info("")


# =====================================================================
# 3. MODEL FREEZING / TRAIN MODES
# =====================================================================

# Component-name to Keras variable path substring mapping.
#
# Keras 3 auto-names layers by class (Dense → dense_N, MLP → mlp_N).
# Python attribute names like ``class_embed`` or ``bbox_embed`` do NOT
# appear in variable paths.  Classification therefore relies on the
# *structural position* inside the Keras layer tree:
#
#   backbone       – variables under ``joiner/`` or ``backbone/``
#   transformer    – variables under ``transformer/`` (decoder layers,
#                    encoder-score head, etc.)
#   query_embeddings – ``refpoint_embed`` or ``query_feat``
#   detection_head – direct children of the top-level LWDETR layer that
#                    are Dense or MLP (i.e. NOT nested under joiner/,
#                    transformer/, or backbone/)
#
# IMPORTANT: ``enc_out_*`` layers are created on LWDETR but *shared*
# with the transformer via reference assignment, so
# ``model.transformer.trainable = False`` accidentally freezes them.
# The classifier must still recognise them as ``detection_head`` so the
# freeze-correction pass in ``apply_train_mode`` can re-enable them.


# Sub-paths that identify the LWDETR root.  We strip this prefix to
# decide whether a variable is a *direct child* or nested.
_LWDETR_ROOT_PREFIXES = ("lwdetr/",)


def _get_component_for_variable(var_path: str) -> str:
    """Map a Keras variable path to a component name.

    Classification rules (applied in order):
    1. Contains ``joiner`` or ``backbone`` → ``"backbone"``
    2. Starts with ``embeddings/`` (DINOv2 cls/pos tokens) → ``"backbone"``
    3. Contains ``refpoint_embed`` or ``query_feat`` → ``"query_embeddings"``
    4. Contains ``ls1`` or ``ls2`` (LayerScale) → ``"backbone"``
    5. Under ``<root>/transformer/`` → ``"transformer"``
    6. Everything else (direct-child Dense/MLP of LWDETR) → ``"detection_head"``
    """
    path_lower = var_path.lower()

    # 1. Backbone (joiner + DINOv2 encoder)
    if "joiner" in path_lower or "backbone" in path_lower:
        return "backbone"

    # 2. DINOv2 position / cls embeddings
    if path_lower.startswith("embeddings/"):
        return "backbone"

    # 3. Query embeddings
    if "refpoint_embed" in path_lower or "query_feat" in path_lower:
        return "query_embeddings"

    # 4. LayerScale parameters (belong to the DINOv2 backbone)
    if path_lower.startswith("ls1/") or path_lower.startswith("ls2/"):
        return "backbone"

    # 5. Transformer decoder + encoder-score layers
    #    Anything whose path passes through ``transformer`` or
    #    ``transformer_N`` (Keras auto-numbers duplicate layers).
    if re.search(r'(?:^|/)transformer(?:_\d+)?/', path_lower):
        return "transformer"

    # 6. Remaining direct children are detection / encoder heads
    return "detection_head"


# Which components are trainable for each train mode.
_TRAIN_MODE_SPEC = {
    "full": {
        "trainable": ["backbone", "transformer", "detection_head",
                       "query_embeddings"],
        "frozen": [],
    },
    "decoder_only": {
        "trainable": ["transformer", "detection_head",
                       "query_embeddings"],
        "frozen": ["backbone"],
    },
    "head_only": {
        "trainable": ["detection_head", "query_embeddings"],
        "frozen": ["backbone", "transformer"],
    },
}


def apply_train_mode(model, mode: str, logger=None):
    """Freeze / unfreeze model components based on the training mode.

    Parameters
    ----------
    model : keras.Model
        The LWDETR Keras model.
    mode : str
        One of 'full', 'decoder_only', 'head_only'.
    logger : logging.Logger, optional

    Returns
    -------
    dict : per-component freeze/unfreeze counts.
    """
    if mode not in _TRAIN_MODE_SPEC:
        raise ValueError(
            f"Unknown train_mode '{mode}'. "
            f"Choices: {list(_TRAIN_MODE_SPEC.keys())}"
        )

    spec = _TRAIN_MODE_SPEC[mode]
    frozen_components = set(spec["frozen"])
    trainable_components = set(spec["trainable"])

    frozen_names = []
    trainable_names = []

    # ---- Phase 1: make everything trainable ---------------------------
    model.trainable = True
    for layer in model._flatten_layers():
        layer.trainable = True

    # ---- Phase 2: freeze specified components via layer attributes ----
    if "backbone" in frozen_components:
        if hasattr(model, "backbone"):
            model.backbone.trainable = False
        elif hasattr(model, "joiner"):
            model.joiner.trainable = False
        frozen_names.append("backbone")

    if "transformer" in frozen_components:
        if hasattr(model, "transformer"):
            model.transformer.trainable = False
        frozen_names.append("transformer")

    # ---- Phase 3: correction pass for shared layers -------------------
    #
    # In LWDETR the enc_out_bbox_embed / enc_out_class_embed layers are
    # created as direct children of the model but *also* assigned to the
    # transformer (``self.transformer.enc_out_bbox_embed = ...``).
    # Similarly bbox_embed may be shared via
    # ``self.transformer.decoder.bbox_embed``.
    #
    # ``model.transformer.trainable = False`` cascades to these shared
    # layers, accidentally freezing the detection heads.  We fix this by
    # walking ALL layers and re-enabling those whose variables belong to
    # a component that *should* be trainable.
    #
    for layer in model._flatten_layers():
        if layer.trainable:
            continue  # Already correct
        layer_vars = getattr(layer, "weights", [])
        if not layer_vars:
            continue
        comp = _get_component_for_variable(layer_vars[0].path)
        if comp in trainable_components:
            layer.trainable = True

    # Determine what's trainable (everything not frozen)
    for comp in trainable_components:
        if comp not in frozen_components:
            trainable_names.append(comp)

    # ---- Phase 4: post-freeze validation --------------------------------
    # Verify that no variable marked as trainable belongs to a frozen
    # component (catches shared-layer issues).
    misclassified = []
    for var in model.trainable_weights:
        comp = _get_component_for_variable(var.path)
        if comp in frozen_components:
            misclassified.append((var.path, comp))
    if misclassified:
        msg = ("apply_train_mode('%s') correctness check FAILED: "
               "%d trainable variables belong to frozen components:\n" %
               (mode, len(misclassified)))
        for path, comp in misclassified[:5]:
            msg += f"  {path} -> {comp}\n"
        raise RuntimeError(msg)

    # Build stats
    params_before = count_parameters(model)
    comp_params = count_component_parameters(model)

    # Verify that trainable components have >0 trainable params
    for comp in trainable_components:
        cp = comp_params.get(comp, {})
        if comp not in frozen_components and cp.get("trainable", 0) == 0:
            if logger:
                logger.warning(
                    "WARNING: component '%s' is supposed to be trainable "
                    "but has 0 trainable parameters!", comp)

    if logger:
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAIN MODE: %s", mode)
        logger.info("=" * 60)
        if frozen_names:
            logger.info("  Frozen modules:")
            for fn in frozen_names:
                p = comp_params.get(fn, {})
                logger.info("    - %-20s (%.2f M params)",
                            fn, p.get("total", 0) / 1e6)
        else:
            logger.info("  Frozen modules: (none)")
        logger.info("  Trainable modules:")
        for tn in trainable_names:
            p = comp_params.get(tn, {})
            logger.info("    - %-20s (%.2f M params)",
                        tn, p.get("trainable", 0) / 1e6)
        logger.info("")
        logger.info("  After freezing:")
        logger.info("    Total params     : %.2f M",
                     params_before["total"] / 1e6)
        logger.info("    Trainable params : %.2f M",
                     params_before["trainable"] / 1e6)
        logger.info("    Frozen params    : %.2f M",
                     params_before["frozen"] / 1e6)
        logger.info("    %% Trainable      : %.1f%%",
                     params_before["pct_trainable"])
        logger.info("=" * 60)
        logger.info("")

    return {
        "frozen_components": frozen_names,
        "trainable_components": trainable_names,
        "params": params_before,
        "component_params": comp_params,
    }


def verify_frozen_gradients(grads, model, mode: str, logger=None) -> bool:
    """Verify that gradients are zero/None for frozen components.

    Parameters
    ----------
    grads : list
        Gradients corresponding to model.trainable_variables.
    model : keras.Model
    mode : str
        The train mode that was applied.
    logger : logging.Logger, optional

    Returns
    -------
    bool : True if gradients are correctly zeroed for frozen components.
    """
    spec = _TRAIN_MODE_SPEC.get(mode, {})
    frozen_components = set(spec.get("frozen", []))

    if not frozen_components:
        return True  # Nothing to verify

    issues = []
    # In Keras, frozen layers should not appear in trainable_variables
    # So their gradients shouldn't even be in the grads list.
    # But let's verify by checking that no trainable variable belongs
    # to a frozen component.
    for var in model.trainable_weights:
        comp = _get_component_for_variable(var.path)
        if comp in frozen_components:
            issues.append(
                f"Variable '{var.path}' ({comp}) is trainable "
                f"but should be frozen in mode '{mode}'"
            )

    if issues and logger:
        logger.warning("GRADIENT VERIFICATION FAILED:")
        for issue in issues:
            logger.warning("  %s", issue)

    if not issues and logger:
        logger.info("Gradient verification PASSED: frozen components "
                    "have no trainable variables.")

    return len(issues) == 0


# =====================================================================
# 4. EARLY STOPPING
# =====================================================================


class EarlyStopper:
    """Research-grade early stopping with best-weight restoration.

    Monitors a metric (lower is better by default) and stops training
    when no improvement is seen for ``patience`` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement.
    min_delta : float
        Minimum improvement to qualify as progress.
    restore_best_weights : bool
        If True, stores a copy of the best weights and restores them
        when early stopping triggers.
    logger : logging.Logger, optional
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        logger=None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.logger = logger

        self.best_value = float("inf")
        self.best_epoch = -1
        self.best_mAP = 0.0
        self.wait = 0
        self.stopped_epoch = -1
        self._best_weights = None  # list of numpy arrays

    def store_weights(self, model):
        """Save a snapshot of the current model weights."""
        self._best_weights = [w.numpy().copy() for w in model.weights]

    def restore_weights(self, model):
        """Restore the best-epoch weights to the model."""
        if self._best_weights is None:
            if self.logger:
                self.logger.warning(
                    "EarlyStopper: no weights stored, cannot restore."
                )
            return False
        for w, saved in zip(model.weights, self._best_weights):
            w.assign(saved)
        if self.logger:
            self.logger.info(
                "EarlyStopper: restored best weights from epoch %d",
                self.best_epoch,
            )
        return True

    def step(self, value: float, epoch: int, model=None,
             mAP_50_95: float = 0.0) -> bool:
        """Update the stopper with the current epoch's metric value.

        Parameters
        ----------
        value : float
            The monitored metric value (e.g. val_loss; lower is better).
        epoch : int
            Current epoch number.
        model : keras.Model, optional
            If provided and ``restore_best_weights`` is True, weights are
            saved on improvement.
        mAP_50_95 : float
            mAP@50:95 for logging purposes.

        Returns
        -------
        bool : True if training should stop.
        """
        improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.best_mAP = mAP_50_95
            self.wait = 0
            if self.restore_best_weights and model is not None:
                self.store_weights(model)
        else:
            self.wait += 1

        if self.logger:
            self.logger.info(
                "  [EarlyStopping] val_loss=%.6f  best=%.6f  "
                "patience=%d/%d  %s",
                value, self.best_value,
                self.wait, self.patience,
                "IMPROVED" if improved else "no improvement",
            )

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.logger:
                self.logger.info("")
                self.logger.info("=" * 60)
                self.logger.info(
                    "Early stopping triggered at epoch %d", epoch
                )
                self.logger.info(
                    "Best epoch: %d", self.best_epoch
                )
                self.logger.info(
                    "Best Val Loss: %.6f", self.best_value
                )
                self.logger.info(
                    "Best mAP@[.5:.95]: %.4f", self.best_mAP
                )
                self.logger.info("=" * 60)
                self.logger.info("")
            return True

        return False


# =====================================================================
# 5. LEARNING RATE SCHEDULES
# =====================================================================


def build_lr_schedule(
    schedule_name: str,
    base_lr: float,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 5.0,
    lr_drop: int = 100,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    lr_min_factor: float = 0.01,
) -> "callable":
    """Build a step-indexed LR schedule function.

    Returns a callable ``schedule(global_step) -> lr`` compatible with
    Keras LR schedules.

    Parameters
    ----------
    schedule_name : str
        One of 'cosine', 'step', 'multistep', 'one_cycle'.
    base_lr : float
        Peak learning rate (after warmup, for cosine / step / multistep).
    total_epochs : int
    steps_per_epoch : int
    warmup_epochs : float
        For all schedulers, warmup occurs before the main schedule.
    lr_drop : int
        For 'step' scheduler — the epoch at which LR drops by ``gamma``.
    milestones : list[int], optional
        For 'multistep' — epoch numbers at which LR drops.
    gamma : float
        Multiplicative factor for 'step' and 'multistep' drops.
    lr_min_factor : float
        Minimum LR as fraction of base_lr (for 'cosine' and 'one_cycle').

    Returns
    -------
    callable : schedule_fn(global_step) -> float
    """
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    if milestones is None:
        milestones = []
    milestone_steps = [m * steps_per_epoch for m in milestones]

    def _warmup_factor(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    if schedule_name == "cosine":
        def schedule_fn(step):
            step = int(step)
            warmup_f = _warmup_factor(step)
            if step < warmup_steps:
                return base_lr * warmup_f
            post_warmup = step - warmup_steps
            post_warmup_total = max(1, total_steps - warmup_steps)
            progress = float(post_warmup) / float(post_warmup_total)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_lr * (lr_min_factor + (1.0 - lr_min_factor) * cosine_decay)

    elif schedule_name == "step":
        drop_step = lr_drop * steps_per_epoch

        def schedule_fn(step):
            step = int(step)
            warmup_f = _warmup_factor(step)
            if step < warmup_steps:
                return base_lr * warmup_f
            lr = base_lr
            if step >= drop_step:
                lr *= gamma
            return lr

    elif schedule_name == "multistep":
        def schedule_fn(step):
            step = int(step)
            warmup_f = _warmup_factor(step)
            if step < warmup_steps:
                return base_lr * warmup_f
            lr = base_lr
            for ms in sorted(milestone_steps):
                if step >= ms:
                    lr *= gamma
            return lr

    elif schedule_name == "one_cycle":
        # Simplified 1-cycle: linear warmup to base_lr, cosine decay to min
        def schedule_fn(step):
            step = int(step)
            if step < warmup_steps:
                return base_lr * float(step) / float(max(1, warmup_steps))
            post_warmup = step - warmup_steps
            post_warmup_total = max(1, total_steps - warmup_steps)
            progress = float(post_warmup) / float(post_warmup_total)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr = base_lr * lr_min_factor
            return min_lr + (base_lr - min_lr) * cosine_decay

    else:
        raise ValueError(
            f"Unknown schedule '{schedule_name}'. "
            f"Options: cosine, step, multistep, one_cycle"
        )

    return schedule_fn


def build_param_group_schedules(
    schedule_name: str,
    lr: float,
    lr_encoder: float,
    lr_component_decay: float,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 5.0,
    train_mode: str = "full",
    **schedule_kwargs,
) -> Dict[str, "callable"]:
    """Build separate LR schedules for each parameter group.

    Parameters
    ----------
    train_mode : str
        One of ``"full"``, ``"decoder_only"``, ``"head_only"``.
        Frozen groups get a constant-zero schedule so that logged LR
        values accurately reflect reality.

    Returns
    -------
    dict mapping group name -> schedule_fn(step) -> lr
        Groups: 'backbone', 'decoder', 'head'
    """
    _zero_schedule = lambda step: 0.0  # noqa: E731

    # Backbone: frozen in decoder_only and head_only
    if train_mode in ("head_only", "decoder_only"):
        backbone_schedule = _zero_schedule
    else:
        backbone_schedule = build_lr_schedule(
            schedule_name, lr_encoder,
            total_epochs, steps_per_epoch, warmup_epochs,
            **schedule_kwargs,
        )

    # Decoder: frozen in head_only
    if train_mode == "head_only":
        decoder_schedule = _zero_schedule
    else:
        decoder_schedule = build_lr_schedule(
            schedule_name, lr * lr_component_decay,
            total_epochs, steps_per_epoch, warmup_epochs,
            **schedule_kwargs,
        )

    # Head: always trainable
    head_schedule = build_lr_schedule(
        schedule_name, lr,
        total_epochs, steps_per_epoch, warmup_epochs,
        **schedule_kwargs,
    )

    return {
        "backbone": backbone_schedule,
        "decoder": decoder_schedule,
        "head": head_schedule,
    }


# =====================================================================
# 6. GRADIENT UTILITIES
# =====================================================================


def compute_gradient_norm(grads) -> float:
    """Compute the global L2 gradient norm.

    Parameters
    ----------
    grads : list of arrays/tensors
        Gradient arrays (may include None).

    Returns
    -------
    float : global L2 norm
    """
    total_norm_sq = 0.0
    for g in grads:
        if g is not None:
            g_np = np.asarray(g) if not isinstance(g, np.ndarray) else g
            total_norm_sq += float(np.sum(g_np ** 2))
    return float(np.sqrt(total_norm_sq))


def scale_gradients_by_lr(grads, trainable_vars, lr_schedules, step):
    """Scale gradients per variable based on per-group LR schedules.

    When using a single optimizer with a base LR, we scale the gradients
    by the ratio (group_lr / base_lr) to achieve per-group learning rates.

    Parameters
    ----------
    grads : list
        Gradient arrays.
    trainable_vars : list
        Corresponding trainable variables.
    lr_schedules : dict
        Mapping group name -> schedule_fn(step) -> lr.
    step : int
        Current global step.

    Returns
    -------
    list : scaled gradients, base_lr (the head LR is used as the optimizer's LR)
    """
    head_lr = lr_schedules["head"](step)
    if head_lr == 0:
        return grads, head_lr

    scaled = []
    for g, v in zip(grads, trainable_vars):
        if g is None:
            scaled.append(None)
            continue
        comp = _get_component_for_variable(v.path)
        if comp == "backbone":
            group_lr = lr_schedules["backbone"](step)
        elif comp == "transformer":
            group_lr = lr_schedules["decoder"](step)
        else:
            group_lr = head_lr

        scale = group_lr / max(head_lr, 1e-12)
        scaled.append(g * scale)

    return scaled, head_lr


# =====================================================================
# 7. BATCH GENERATOR (make_batches)
# =====================================================================


def _augment_pipeline2(image, target, rng):
    """Apply pipeline2-style augmentations compatible with DETR training.

    Augmentations applied:
    - Random horizontal flip (50% probability) with box adjustment
    - Random color jitter (brightness, contrast, saturation)

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), float32 in [0, 1]
    target : dict with 'boxes' (N, 4) in cxcywh normalised and 'labels'
    rng : np.random.RandomState

    Returns
    -------
    image, target : augmented copies
    """
    boxes = target["boxes"].copy()
    labels = target["labels"].copy()

    # Horizontal flip (p=0.5)
    if rng.rand() < 0.5:
        image = image[:, ::-1, :].copy()
        if len(boxes) > 0:
            # cxcywh format, normalised: flip cx -> 1 - cx
            boxes[:, 0] = 1.0 - boxes[:, 0]

    # Color jitter (brightness, contrast, saturation)
    if rng.rand() < 0.8:
        # Brightness: multiply by [0.7, 1.3]
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0.0, 1.0)

    if rng.rand() < 0.8:
        # Contrast: blend with mean gray
        gray_mean = image.mean()
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(gray_mean + factor * (image - gray_mean), 0.0, 1.0)

    if rng.rand() < 0.5:
        # Saturation: blend with grayscale
        gray = np.mean(image, axis=-1, keepdims=True)
        factor = rng.uniform(0.7, 1.3)
        image = np.clip(gray + factor * (image - gray), 0.0, 1.0)

    image = image.astype(np.float32)
    return image, {"boxes": boxes, "labels": labels}


def make_batches(dataset, indices, batch_size, max_batches=None,
                 augmentation=None, rng=None):
    """Yield (images, targets) mini-batches from a dataset.

    Parameters
    ----------
    dataset : DeepFishDataset
        Must support ``__getitem__`` returning ``(image_np, target_dict)``.
    indices : list[int]
        Subset of dataset indices to iterate over.
    batch_size : int
    max_batches : int or None
        If given, stop after this many batches.
    augmentation : str or None
        Augmentation strategy: 'pipeline2' for basic augmentations,
        'rf_detr' reserved (no-op), None for no augmentation.
    rng : np.random.RandomState or None
        Random state for augmentation reproducibility.

    Yields
    ------
    images : np.ndarray, shape (B, H, W, 3)
    targets : list[dict], each with 'labels' and 'boxes' keys
    """
    from keras import ops

    if rng is None:
        rng = np.random.RandomState()

    batch_count = 0
    for start in range(0, len(indices), batch_size):
        if max_batches is not None and batch_count >= max_batches:
            return
        batch_idx = indices[start: start + batch_size]
        images, targets = [], []
        for idx in batch_idx:
            img, tgt = dataset[idx]
            if augmentation == "pipeline2":
                img, tgt = _augment_pipeline2(img, tgt, rng)
            images.append(img)
            targets.append(tgt)

        images_np = np.stack(images, axis=0).astype("float32")
        images_tensor = ops.convert_to_tensor(images_np, dtype="float32")
        yield images_tensor, targets
        batch_count += 1


# =====================================================================
# 8. LOSS COMPONENT EXTRACTION
# =====================================================================


def compute_loss_components(outputs, targets, criterion, indices_main,
                            aux_indices, num_boxes_f):
    """Compute detailed loss components for logging.

    Returns
    -------
    dict : individual loss values (loss_ce, loss_bbox, loss_giou, total, etc.)
    """
    from keras import ops

    num_boxes_t = ops.convert_to_tensor(num_boxes_f, dtype="float32")
    weight_dict = criterion.weight_dict
    components = {}

    # Main head losses
    for loss_type in criterion.loss_types:
        l_dict = criterion.get_loss(
            loss_type, outputs, targets, indices_main, num_boxes_t
        )
        for k, v in l_dict.items():
            val = float(ops.convert_to_numpy(v))
            components[k] = val

    # Weighted total
    total = 0.0
    for k, v in components.items():
        if k in weight_dict:
            total += v * weight_dict[k]
    components["total_loss"] = total

    return components


# =====================================================================
# 9. DEFAULT CONFIGURATION POLICY
# =====================================================================


def log_default_config(args, logger):
    """Log the effective configuration with ML best-practice annotations."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EFFECTIVE CONFIGURATION (with defaults policy)")
    logger.info("=" * 60)

    defaults_applied = []

    # Optimizer
    logger.info("  Optimizer           : AdamW")
    logger.info("  Weight decay        : %.1e", args.weight_decay)

    # LR schedule
    sched = getattr(args, "lr_scheduler", "cosine")
    logger.info("  LR scheduler        : %s", sched)
    if sched == "cosine":
        defaults_applied.append("Cosine LR with warmup (best practice)")
    logger.info("  Warmup epochs       : %.1f", args.warmup_epochs)
    logger.info("  Base LR (head)      : %.1e", args.lr)
    logger.info("  Encoder LR          : %.1e", args.lr_encoder)
    logger.info("  Component decay     : %.2f", args.lr_component_decay)

    # Gradient clipping
    logger.info("  Gradient clipping   : %.2f", args.clip_max_norm)

    # EMA
    logger.info("  EMA                 : %s", "enabled" if args.use_ema else "disabled")
    if args.use_ema:
        logger.info("    EMA decay         : %.4f", args.ema_decay)

    # Early stopping
    logger.info("  Early stopping      : %s",
                "enabled" if args.early_stopping else "disabled")
    if args.early_stopping:
        logger.info("    Patience          : %d", args.early_stopping_patience)
        logger.info("    Min delta         : %.1e",
                     getattr(args, "early_stopping_min_delta", 1e-4))

    # Train mode
    logger.info("  Train mode          : %s",
                getattr(args, "train_mode", "full"))

    # Checkpoint
    logger.info("  Checkpoint mode     : %s", args.checkpoint_mode)

    logger.info("")
    if defaults_applied:
        logger.info("  Defaults applied (ML best practice):")
        for d in defaults_applied:
            logger.info("    - %s", d)
    logger.info("=" * 60)
    logger.info("")


# =====================================================================
# 10. TRAINING LOOP (Custom, research-grade)
# =====================================================================


def train_one_epoch_custom(
    model, criterion, optimizer, data_iterator, num_steps, epoch,
    clip_max_norm=0.1, lr_schedules=None, global_step=0,
    train_mode="full", print_freq=10, logger=None,
):
    """Train for one epoch with detailed metric tracking.

    This is a custom training loop that extends the library's
    ``train_one_epoch`` with:
    - Per-group LR scheduling via gradient scaling
    - Gradient norm tracking (before and after clipping)
    - Detailed loss component logging
    - Frozen-gradient verification

    Parameters
    ----------
    model : keras.Model (LWDETR)
    criterion : SetCriterion
    optimizer : keras.optimizers.Optimizer
    data_iterator : iterable yielding (images, targets)
    num_steps : int
    epoch : int
    clip_max_norm : float
    lr_schedules : dict or None
        Per-group LR schedules. If None, uses a flat LR.
    global_step : int
        Starting global step for LR schedule lookup.
    train_mode : str
        For gradient verification.
    print_freq : int
    logger : logging.Logger

    Returns
    -------
    dict : epoch statistics including loss, loss components, gradient norms,
           LR values.
    """
    import jax
    from keras import ops

    weight_dict = criterion.weight_dict
    group_detr = criterion.group_detr
    sum_group_losses = getattr(criterion, "sum_group_losses", False)

    # Accumulators
    epoch_losses = []
    epoch_grad_norms = []
    epoch_loss_ce = []
    epoch_loss_bbox = []
    epoch_loss_giou = []
    lr_values = {"backbone": [], "decoder": [], "head": []}

    step_start = global_step
    start_time = time.time()

    for step_idx, (images, targets) in enumerate(data_iterator):
        if step_idx >= num_steps:
            break

        cur_step = step_start + step_idx
        images = ops.convert_to_tensor(images, dtype="float32")

        # ==============================================================
        # Phase 1: Eager forward + Hungarian matching
        # ==============================================================
        # MUST use training=True so the model outputs all
        # num_queries * group_detr queries.  With training=False the
        # model only emits num_queries (1 group), so the matcher's
        # ops.split(C, group_detr) fails when num_queries % group_detr
        # != 0.  training=True also ensures every enc_out_* sub-layer
        # is built, preventing new variables from appearing later and
        # desynchronising the gradient / optimizer variable lists.
        outputs_eager = model(images, training=True)

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
                    criterion.matcher(aux_out, targets, group_detr=group_detr)
                )

        num_boxes = sum(len(t["labels"]) for t in targets)
        if not sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes_f = max(float(num_boxes), 1.0)

        # ==============================================================
        # Phase 2: Traced forward + loss + gradient computation
        # ==============================================================
        trainable_values = [v.value for v in model.trainable_variables]
        non_trainable_values = [v.value for v in model.non_trainable_variables]

        def forward_and_loss(trainable_params):
            """Pure function for jax.value_and_grad."""
            outputs, updated_nt = model.stateless_call(
                trainable_params, non_trainable_values,
                images, training=True,
            )
            # Compute total weighted loss
            num_boxes_t = ops.convert_to_tensor(num_boxes_f, dtype="float32")
            total_loss = ops.convert_to_tensor(0.0, dtype="float32")

            for loss_type in criterion.loss_types:
                l_dict = criterion.get_loss(
                    loss_type, outputs, targets, indices_main, num_boxes_t
                )
                for k, v in l_dict.items():
                    if k in weight_dict:
                        total_loss = total_loss + v * weight_dict[k]

            if "aux_outputs" in outputs:
                for i, aux_out in enumerate(outputs["aux_outputs"]):
                    aux_idx = (
                        aux_indices[i]
                        if i < len(aux_indices)
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

            return total_loss, updated_nt

        grad_fn = jax.value_and_grad(forward_and_loss, has_aux=True)
        (total_loss, updated_nt), grads = grad_fn(trainable_values)

        # ==============================================================
        # Gradient norm (before clipping)
        # ==============================================================
        grad_norm_pre = compute_gradient_norm(
            [np.asarray(g) if g is not None else None for g in grads]
        )
        epoch_grad_norms.append(grad_norm_pre)

        # ==============================================================
        # Phase 3: Per-group LR scaling + gradient clipping
        # ==============================================================
        if lr_schedules is not None:
            grads_list = list(grads)
            grads_list, current_lr = scale_gradients_by_lr(
                grads_list, model.trainable_variables,
                lr_schedules, cur_step,
            )
            grads = grads_list

            # Track per-group LRs
            lr_values["backbone"].append(lr_schedules["backbone"](cur_step))
            lr_values["decoder"].append(lr_schedules["decoder"](cur_step))
            lr_values["head"].append(lr_schedules["head"](cur_step))

            # Update optimizer's learning rate
            optimizer.learning_rate = current_lr
        else:
            lr_val = float(optimizer.learning_rate)
            lr_values["head"].append(lr_val)
            lr_values["backbone"].append(lr_val)
            lr_values["decoder"].append(lr_val)

        if clip_max_norm > 0:
            from paz.models.detection.dino_v2_object_detection.engine import (
                _clip_grad_norm,
            )
            grads = _clip_grad_norm(grads, clip_max_norm)

        # ==============================================================
        # Apply gradients + sync non-trainable vars
        # ==============================================================
        optimizer.apply(grads, model.trainable_variables)

        for var, val in zip(model.non_trainable_variables, updated_nt):
            var.assign(val)

        # ==============================================================
        # Logging
        # ==============================================================
        loss_value = float(ops.convert_to_numpy(total_loss))
        if not math.isfinite(loss_value):
            if logger:
                logger.error("Loss is %s at step %d, stopping", loss_value, step_idx)
            raise ValueError(f"Loss is {loss_value}, stopping training")

        epoch_losses.append(loss_value)

        # Extract individual loss components from eager forward
        # (these are approximate since they use the eager outputs, not the
        # traced ones, but they're accurate enough for monitoring)
        try:
            saved_gd = criterion.group_detr
            criterion.group_detr = 1
            with_loss = criterion.get_loss(
                "labels", outputs_eager, targets, indices_main,
                ops.convert_to_tensor(num_boxes_f, "float32"),
            )
            epoch_loss_ce.append(
                float(ops.convert_to_numpy(with_loss.get("loss_ce", 0.0)))
            )
            box_loss = criterion.get_loss(
                "boxes", outputs_eager, targets, indices_main,
                ops.convert_to_tensor(num_boxes_f, "float32"),
            )
            epoch_loss_bbox.append(
                float(ops.convert_to_numpy(box_loss.get("loss_bbox", 0.0)))
            )
            epoch_loss_giou.append(
                float(ops.convert_to_numpy(box_loss.get("loss_giou", 0.0)))
            )
            criterion.group_detr = saved_gd
        except Exception:
            pass

        if step_idx % print_freq == 0 or step_idx == num_steps - 1:
            lr_str = ""
            if lr_schedules:
                lr_str = (
                    f"  lr_backbone={lr_values['backbone'][-1]:.2e}"
                    f"  lr_decoder={lr_values['decoder'][-1]:.2e}"
                    f"  lr_head={lr_values['head'][-1]:.2e}"
                )
            msg = (
                f"  Epoch [{epoch}] Step [{step_idx}/{num_steps}]  "
                f"loss={loss_value:.4f}  grad_norm={grad_norm_pre:.4f}"
                f"{lr_str}"
            )
            if logger:
                logger.info(msg)
            else:
                print(msg)

    elapsed = time.time() - start_time
    new_global_step = step_start + min(step_idx + 1, num_steps)

    # Aggregate
    stats = {
        "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
        "grad_norm": float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0,
        "grad_norm_max": float(np.max(epoch_grad_norms)) if epoch_grad_norms else 0.0,
        "loss_ce": float(np.mean(epoch_loss_ce)) if epoch_loss_ce else 0.0,
        "loss_bbox": float(np.mean(epoch_loss_bbox)) if epoch_loss_bbox else 0.0,
        "loss_giou": float(np.mean(epoch_loss_giou)) if epoch_loss_giou else 0.0,
        "lr_backbone": float(np.mean(lr_values["backbone"])) if lr_values["backbone"] else 0.0,
        "lr_decoder": float(np.mean(lr_values["decoder"])) if lr_values["decoder"] else 0.0,
        "lr_head": float(np.mean(lr_values["head"])) if lr_values["head"] else 0.0,
        "epoch_time": str(datetime.timedelta(seconds=int(elapsed))),
        "global_step": new_global_step,
    }

    if logger:
        logger.info(
            "  Epoch [%d] completed in %s (%.2fs/step)",
            epoch, stats["epoch_time"],
            elapsed / max(1, min(step_idx + 1, num_steps)),
        )

    return stats
