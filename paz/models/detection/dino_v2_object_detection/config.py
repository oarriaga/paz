from collections import namedtuple


ModelConfig = namedtuple("ModelConfig", [
    "encoder", "out_feature_indexes", "dec_layers", "two_stage",
    "projector_scale", "hidden_dim", "patch_size", "num_windows",
    "sa_nheads", "ca_nheads", "dec_n_points", "bbox_reparam",
    "lite_refpoint_refine", "layer_norm", "num_classes",
    "pretrain_weights", "resolution", "group_detr",
    "positional_encoding_size", "ia_bce_loss", "cls_loss_coef",
    "segmentation_head", "mask_downsample_ratio", "num_queries",
    "num_select",
], defaults=[
    "dinov2_windowed_small", [2, 5, 8, 11], 3, True, ["P4"], 256, 14, 4,
    8, 16, 2, True, True, True, 90, None, 560, 13, 37, True, 1.0, False,
    4, 300, 300,
])


# ---- Detection variants ------------------------------------------------


def RFDETRBaseConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_small", hidden_dim=256, patch_size=14,
        num_windows=4, dec_layers=3, sa_nheads=8, ca_nheads=16,
        dec_n_points=2, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[1, 4, 7, 10],
        pretrain_weights="lwdetr_base.weights.h5", resolution=560,
        positional_encoding_size=37,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRNanoConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_small", hidden_dim=256, patch_size=16,
        num_windows=2, dec_layers=2, sa_nheads=8, ca_nheads=16,
        dec_n_points=2, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="lwdetr_nano.weights.h5", resolution=384,
        positional_encoding_size=24,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSmallConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_small", hidden_dim=256, patch_size=16,
        num_windows=2, dec_layers=3, sa_nheads=8, ca_nheads=16,
        dec_n_points=2, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="lwdetr_small.weights.h5", resolution=512,
        positional_encoding_size=32,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRMediumConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_small", hidden_dim=256, patch_size=16,
        num_windows=2, dec_layers=4, sa_nheads=8, ca_nheads=16,
        dec_n_points=2, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="lwdetr_medium.weights.h5", resolution=576,
        positional_encoding_size=36,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRLargeConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_small", hidden_dim=256, patch_size=16,
        num_windows=2, dec_layers=4, sa_nheads=8, ca_nheads=16,
        dec_n_points=2, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="lwdetr_large.weights.h5", resolution=704,
        positional_encoding_size=44,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRXLargeConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_base", hidden_dim=512, patch_size=20,
        num_windows=1, dec_layers=5, sa_nheads=16, ca_nheads=32,
        dec_n_points=4, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[2, 5, 8, 11],
        num_classes=365, pretrain_weights="lwdetr_xlarge.weights.h5",
        resolution=700, positional_encoding_size=35,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETR2XLargeConfig(**kwargs):
    base = dict(
        encoder="dinov2_windowed_base", hidden_dim=512, patch_size=20,
        num_windows=2, dec_layers=5, sa_nheads=16, ca_nheads=32,
        dec_n_points=4, num_queries=300, num_select=300,
        projector_scale=["P4"], out_feature_indexes=[2, 5, 8, 11],
        num_classes=365, pretrain_weights="lwdetr_2xlarge.weights.h5",
        resolution=880, positional_encoding_size=44,
    )
    return ModelConfig(**{**base, **kwargs})


# ---- Segmentation variants ---------------------------------------------


def RFDETRSegPreviewConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=2, dec_layers=4,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=200,
        num_select=200, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-preview.pt", resolution=432,
        positional_encoding_size=36,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSegNanoConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=1, dec_layers=4,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=100,
        num_select=100, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-nano.pt", resolution=312,
        positional_encoding_size=26,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSegSmallConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=2, dec_layers=4,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=100,
        num_select=100, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-small.pt", resolution=384,
        positional_encoding_size=32,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSegMediumConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=2, dec_layers=5,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=200,
        num_select=200, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-medium.pt", resolution=432,
        positional_encoding_size=36,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSegLargeConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=2, dec_layers=5,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=200,
        num_select=200, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-large.pt", resolution=504,
        positional_encoding_size=42,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSegXLargeConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=2, dec_layers=6,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=300,
        num_select=300, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-xlarge.pt", resolution=624,
        positional_encoding_size=52,
    )
    return ModelConfig(**{**base, **kwargs})


def RFDETRSeg2XLargeConfig(**kwargs):
    base = dict(
        segmentation_head=True, encoder="dinov2_windowed_small",
        hidden_dim=256, patch_size=12, num_windows=2, dec_layers=6,
        sa_nheads=8, ca_nheads=16, dec_n_points=2, num_queries=300,
        num_select=300, projector_scale=["P4"],
        out_feature_indexes=[2, 5, 8, 11],
        pretrain_weights="rf-detr-seg-xxlarge.pt", resolution=768,
        positional_encoding_size=64,
    )
    return ModelConfig(**{**base, **kwargs})


# ---- Training configs ---------------------------------------------------


TrainConfig = namedtuple("TrainConfig", [
    "lr", "lr_encoder", "batch_size", "grad_accum_steps", "epochs",
    "ema_decay", "ema_tau", "lr_drop", "checkpoint_interval",
    "warmup_epochs", "lr_vit_layer_decay", "lr_component_decay",
    "lr_scheduler", "lr_min_factor", "drop_path", "dropout", "group_detr",
    "ia_bce_loss", "cls_loss_coef", "dataset_file", "square_resize_div_64",
    "dataset_dir", "output_dir", "multi_scale", "expanded_scales",
    "do_random_resize_via_padding", "use_ema", "num_workers",
    "weight_decay", "early_stopping", "early_stopping_patience",
    "early_stopping_min_delta", "early_stopping_use_ema", "tensorboard",
    "wandb", "project", "run", "class_names", "run_test", "clip_max_norm",
    "segmentation_head", "eval_max_dets", "resume", "amp", "fp16_eval",
    "backbone_lora", "lora_rank", "lora_alpha", "use_dora",
], defaults=[
    1e-4, 1.5e-4, 4, 4, 100, 0.993, 100, 100, 10, 0.0, 0.8, 0.7, "step",
    0.0, 0.0, 0.0, 13, True, 1.0, "coco_json", True, "", "output", True,
    True, False, True, 2, 1e-4, False, 10, 0.001, False, False, False,
    None, None, None, True, 0.1, False, 500, False, True, False, False,
    16, 16, True,
])


# SegmentationTrainConfig adds mask fields and overrides the two coefs the
# reference sets for segmentation, mirroring the previous subclass.
_SEG_TRAIN_FIELDS = TrainConfig._fields + (
    "mask_point_sample_ratio", "mask_ce_loss_coef", "mask_dice_loss_coef",
)
_SEG_TRAIN_DEFAULTS = {
    **TrainConfig._field_defaults,
    "cls_loss_coef": 5.0,
    "segmentation_head": True,
    "mask_point_sample_ratio": 16,
    "mask_ce_loss_coef": 5.0,
    "mask_dice_loss_coef": 5.0,
}
SegmentationTrainConfig = namedtuple(
    "SegmentationTrainConfig", _SEG_TRAIN_FIELDS,
    defaults=[_SEG_TRAIN_DEFAULTS[name] for name in _SEG_TRAIN_FIELDS],
)


# ---- Registry (all config builders, keyed by name) ----------------------

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
