from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Base configuration shared by every RF-DETR variant."""

    encoder: str = "dinov2_windowed_small"
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    dec_layers: int = 3
    two_stage: bool = True
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    resolution: int = 560
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int = 37
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    num_queries: int = 300
    num_select: int = 300


# ---- Detection variants ------------------------------------------------


@dataclass
class RFDETRBaseConfig(ModelConfig):
    """RF-DETR Base (29 M params)."""

    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [1, 4, 7, 10])
    pretrain_weights: Optional[str] = "lwdetr_base.weights.h5"
    resolution: int = 560
    positional_encoding_size: int = 37  # matches PyTorch original DINOv2 default


@dataclass
class RFDETRNanoConfig(ModelConfig):
    """RF-DETR Nano."""

    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 2
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "lwdetr_nano.weights.h5"
    resolution: int = 384
    positional_encoding_size: int = 24


@dataclass
class RFDETRSmallConfig(ModelConfig):
    """RF-DETR Small."""

    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "lwdetr_small.weights.h5"
    resolution: int = 512
    positional_encoding_size: int = 32


@dataclass
class RFDETRMediumConfig(ModelConfig):
    """RF-DETR Medium."""

    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "lwdetr_medium.weights.h5"
    resolution: int = 576
    positional_encoding_size: int = 36


@dataclass
class RFDETRLargeConfig(ModelConfig):
    """RF-DETR Large (2026)."""

    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "lwdetr_large.weights.h5"
    resolution: int = 704
    positional_encoding_size: int = 44  # 704 // 16


@dataclass
class RFDETRXLargeConfig(ModelConfig):
    """RF-DETR XLarge."""

    encoder: str = "dinov2_windowed_base"
    hidden_dim: int = 512
    patch_size: int = 20
    num_windows: int = 1
    dec_layers: int = 5
    sa_nheads: int = 16
    ca_nheads: int = 32
    dec_n_points: int = 4
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    num_classes: int = 365
    pretrain_weights: Optional[str] = "lwdetr_xlarge.weights.h5"
    resolution: int = 700
    positional_encoding_size: int = 35  # 700 // 20


@dataclass
class RFDETR2XLargeConfig(ModelConfig):
    """RF-DETR 2XLarge."""

    encoder: str = "dinov2_windowed_base"
    hidden_dim: int = 512
    patch_size: int = 20
    num_windows: int = 2
    dec_layers: int = 5
    sa_nheads: int = 16
    ca_nheads: int = 32
    dec_n_points: int = 4
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    num_classes: int = 365
    pretrain_weights: Optional[str] = "lwdetr_2xlarge.weights.h5"
    resolution: int = 880
    positional_encoding_size: int = 44  # 880 // 20


# ---- Segmentation variants ---------------------------------------------


@dataclass
class RFDETRSegPreviewConfig(ModelConfig):
    """RF-DETR Segmentation Preview."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 2
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 200
    num_select: int = 200
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    resolution: int = 432
    positional_encoding_size: int = 36


@dataclass
class RFDETRSegNanoConfig(ModelConfig):
    """RF-DETR Segmentation Nano."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 1
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 100
    num_select: int = 100
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-nano.pt"
    resolution: int = 312
    positional_encoding_size: int = 26  # 312 // 12


@dataclass
class RFDETRSegSmallConfig(ModelConfig):
    """RF-DETR Segmentation Small."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 2
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 100
    num_select: int = 100
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-small.pt"
    resolution: int = 384
    positional_encoding_size: int = 32  # 384 // 12


@dataclass
class RFDETRSegMediumConfig(ModelConfig):
    """RF-DETR Segmentation Medium."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 2
    dec_layers: int = 5
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 200
    num_select: int = 200
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-medium.pt"
    resolution: int = 432
    positional_encoding_size: int = 36  # 432 // 12


@dataclass
class RFDETRSegLargeConfig(ModelConfig):
    """RF-DETR Segmentation Large."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 2
    dec_layers: int = 5
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 200
    num_select: int = 200
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-large.pt"
    resolution: int = 504
    positional_encoding_size: int = 42  # 504 // 12


@dataclass
class RFDETRSegXLargeConfig(ModelConfig):
    """RF-DETR Segmentation XLarge."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 2
    dec_layers: int = 6
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-xlarge.pt"
    resolution: int = 624
    positional_encoding_size: int = 52  # 624 // 12


@dataclass
class RFDETRSeg2XLargeConfig(ModelConfig):
    """RF-DETR Segmentation 2XLarge."""

    segmentation_head: bool = True
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 12
    num_windows: int = 2
    dec_layers: int = 6
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    pretrain_weights: Optional[str] = "rf-detr-seg-xxlarge.pt"
    resolution: int = 768
    positional_encoding_size: int = 64  # 768 // 12


# ---- Training configs ---------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyper-parameters (detection)."""

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
    dataset_file: str = "coco_json"
    square_resize_div_64: bool = True
    dataset_dir: str = ""
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
    tensorboard: bool = False
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: Optional[List[str]] = None
    run_test: bool = True
    clip_max_norm: float = 0.1
    segmentation_head: bool = False
    eval_max_dets: int = 500


@dataclass
class SegmentationTrainConfig(TrainConfig):
    """Training hyper-parameters (segmentation)."""

    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True


# ---- Registry (all config classes, keyed by name) -----------------------

MODEL_CONFIG_REGISTRY = {
    "RFDETRBase": RFDETRBaseConfig,
    "RFDETRNano": RFDETRNanoConfig,
    "RFDETRSmall": RFDETRSmallConfig,
    "RFDETRMedium": RFDETRMediumConfig,
    "RFDETRLarge": RFDETRLargeConfig,
    "RFDETRXLarge": RFDETRXLargeConfig,
    "RFDETR2XLarge": RFDETR2XLargeConfig,
    "RFDETRSegPreview": RFDETRSegPreviewConfig,
    "RFDETRSegNano": RFDETRSegNanoConfig,
    "RFDETRSegSmall": RFDETRSegSmallConfig,
    "RFDETRSegMedium": RFDETRSegMediumConfig,
    "RFDETRSegLarge": RFDETRSegLargeConfig,
    "RFDETRSegXLarge": RFDETRSegXLargeConfig,
    "RFDETRSeg2XLarge": RFDETRSeg2XLargeConfig,
}
