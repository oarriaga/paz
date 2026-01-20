import os
import sys

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
# strict constraint: set JAX backend before importing Keras
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from dataclasses import dataclass, field, is_dataclass
from typing import List, Optional, Literal, Union, get_args, get_origin, get_type_hints


# -----------------------------------------------------------------------------
# 2. VALIDATION UTILITIES (Pydantic Replacement)
# -----------------------------------------------------------------------------
def _check_type(value, type_hint, field_name):
    """
    Recursively validates a value against a type hint.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Optional (Union[T, None])
    if origin is Union and type(None) in args:
        if value is None:
            return  # Valid
        # Extract the non-None type
        actual_types = [t for t in args if t is not type(None)]
        if len(actual_types) == 1:
            _check_type(value, actual_types[0], field_name)
        return

    # Handle Literal
    if origin is Literal:
        if value not in args:
            raise ValueError(
                f"Field '{field_name}' got '{value}', expected one of {args}"
            )
        return

    # Handle List
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Field '{field_name}' expected a list, got {type(value).__name__}"
            )

        # If List[T], validate items
        if args:
            inner_type = args[0]
            for i, item in enumerate(value):
                _check_type(item, inner_type, f"{field_name}[{i}]")
        return

    # Handle Standard Types (int, float, str, bool)
    if isinstance(type_hint, type):
        # Allow float for int fields if they are whole numbers (Pydantic behaviorish)
        # but strict type checking is safer. Here we stick to instance checks.
        if type_hint is float and isinstance(value, int):
            return  # valid
        if not isinstance(value, type_hint):
            raise TypeError(
                f"Field '{field_name}' expected {type_hint.__name__}, got {type(value).__name__}"
            )


def validate_config(instance):
    """
    Validates the dataclass fields against their type hints.
    """
    # get_type_hints is safer than __annotations__ for handling forward refs/imports
    hints = get_type_hints(instance.__class__)

    for field_name, field_type in hints.items():
        # Skip internal fields if any
        if field_name.startswith("_"):
            continue

        value = getattr(instance, field_name)

        # Validation Logic
        try:
            _check_type(value, field_type, field_name)
        except (TypeError, ValueError) as e:
            # Re-raise with cleaner context if needed, or just let it bubble
            raise e


# -----------------------------------------------------------------------------
# 3. MODEL CONFIGURATIONS
# -----------------------------------------------------------------------------


@dataclass(kw_only=True)
class ModelConfig:
    mask_downsample_ratio: int = 4

    def __post_init__(self):
        validate_config(self)


@dataclass(kw_only=True)
class RFDETRBaseConfig(ModelConfig):
    positional_encoding_size: int = 37


@dataclass(kw_only=True)
class RFDETRLargeConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Large model."""

    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = (
        "dinov2_windowed_base"
    )
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = field(
        default_factory=lambda: ["P3", "P5"]
    )
    pretrain_weights: Optional[str] = "rf-detr-large.pth"


@dataclass(kw_only=True)
class RFDETRNanoConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Nano model."""

    out_feature_indexes: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"


@dataclass(kw_only=True)
class RFDETRSmallConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Small model."""

    out_feature_indexes: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"


@dataclass(kw_only=True)
class RFDETRMediumConfig(RFDETRBaseConfig):
    """The configuration for an RF-DETR Medium model."""

    out_feature_indexes: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"


@dataclass(kw_only=True)
class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90


# -----------------------------------------------------------------------------
# 4. TRAINING CONFIGURATIONS
# -----------------------------------------------------------------------------


@dataclass(kw_only=True)
class TrainConfig:
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 100
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 10
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True

    # REQUIRED field (no default)
    dataset_dir: str

    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: Optional[List[str]] = None
    run_test: bool = True
    segmentation_head: bool = False

    def __post_init__(self):
        validate_config(self)


@dataclass(kw_only=True)
class SegmentationTrainConfig(TrainConfig):
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True
