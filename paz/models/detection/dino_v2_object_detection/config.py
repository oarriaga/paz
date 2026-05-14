from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Base architecture configuration shared by every RF-DETR variant.

    Attributes:
        encoder (str): DINOv2 backbone variant name.
        out_feature_indexes (List[int]): Backbone block indices to tap.
        dec_layers (int): Number of transformer decoder layers.
        two_stage (bool): Enable two-stage (encoder proposal) mode.
        projector_scale (List[str]): Feature pyramid level names.
        hidden_dim (int): Transformer hidden dimension.
        patch_size (int): ViT patch size.
        num_windows (int): Number of attention windows in the backbone.
        sa_nheads (int): Self-attention head count.
        ca_nheads (int): Cross-attention head count.
        dec_n_points (int): Deformable attention sampling points.
        bbox_reparam (bool): Use bounding-box reparameterisation.
        lite_refpoint_refine (bool): Lightweight reference-point refinement.
        layer_norm (bool): Apply layer normalisation in the projector.
        num_classes (int): Number of object categories (excluding background).
        pretrain_weights (Optional[str]): Filename of pretrained weights.
        resolution (int): Input image resolution (square).
        group_detr (int): GROUP-DETR query duplication factor.
        gradient_checkpointing (bool): Enable gradient checkpointing.
        positional_encoding_size (int): Spatial size of positional encoding.
        ia_bce_loss (bool): Use instance-aware BCE loss.
        cls_loss_coef (float): Classification loss coefficient.
        segmentation_head (bool): Attach a segmentation head.
        mask_downsample_ratio (int): Mask spatial downsample ratio.
        num_queries (int): Total number of object queries.
        num_select (int): Queries selected for post-processing.
    """

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
    positional_encoding_size: int = 37  # DINOv2 default for patch_size=14


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
    """Training hyperparameters for detection.

    Attributes:
        lr (float): Base learning rate.
        lr_encoder (float): Learning rate for the backbone encoder.
        batch_size (int): Images per device per step.
        grad_accum_steps (int): Gradient accumulation steps.
        epochs (int): Total training epochs.
        ema_decay (float): Exponential moving average decay.
        ema_tau (int): EMA warm-up steps.
        lr_drop (int): Epoch at which to drop the LR (step schedule).
        checkpoint_interval (int): Save a checkpoint every N epochs.
        warmup_epochs (float): Linear warm-up duration in epochs.
        lr_vit_layer_decay (float): Per-layer LR decay for the ViT.
        lr_component_decay (float): Component-wise LR decay.
        drop_path (float): Stochastic depth rate.
        group_detr (int): GROUP-DETR duplication factor.
        ia_bce_loss (bool): Instance-aware BCE classification loss.
        cls_loss_coef (float): Classification loss weight.
        dataset_file (str): Dataset format (``coco_json`` or ``coco``).
        square_resize_div_64 (bool): Resize to nearest multiple of 64.
        dataset_dir (str): Root directory of the dataset.
        output_dir (str): Directory for checkpoints and logs.
        multi_scale (bool): Multi-scale data augmentation.
        expanded_scales (bool): Extended scale range.
        do_random_resize_via_padding (bool): Pad-based random resize.
        use_ema (bool): Use exponential moving average.
        num_workers (int): Data-loader worker count.
        weight_decay (float): AdamW weight decay.
        early_stopping (bool): Enable early stopping.
        early_stopping_patience (int): Patience (epochs) for early stop.
        early_stopping_min_delta (float): Minimum improvement delta.
        early_stopping_use_ema (bool): Monitor EMA metric for early stop.
        tensorboard (bool): Log to TensorBoard.
        wandb (bool): Log to Weights & Biases.
        project (Optional[str]): W&B project name.
        run (Optional[str]): W&B run name.
        class_names (Optional[List[str]]): Class label names.
        run_test (bool): Run evaluation after training.
        clip_max_norm (float): Max gradient norm for clipping.
        segmentation_head (bool): Train with segmentation head.
        eval_max_dets (int): Maximum detections per image at evaluation.
    """

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
    lr_scheduler: str = "step"
    lr_min_factor: float = 0.0
    drop_path: float = 0.0
    dropout: float = 0.0
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
    resume: bool = False
    amp: bool = True
    fp16_eval: bool = False
    backbone_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    use_dora: bool = True


@dataclass
class SegmentationTrainConfig(TrainConfig):
    """Training hyperparameters for instance segmentation.

    Extends ``TrainConfig`` with mask-specific loss weights and enables
    the segmentation head by default.

    Attributes:
        mask_point_sample_ratio (int): Points per mask for point-based loss.
        mask_ce_loss_coef (float): Mask cross-entropy loss weight.
        mask_dice_loss_coef (float): Mask Dice loss weight.
        cls_loss_coef (float): Classification loss weight (overridden).
        segmentation_head (bool): Always ``True``.
    """

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
