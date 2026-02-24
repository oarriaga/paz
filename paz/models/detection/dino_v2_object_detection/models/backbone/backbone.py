import keras
from keras import layers, ops

from paz.models.detection.dino_v2_object_detection.models.backbone.dinov2 import DinoV2
from paz.models.detection.dino_v2_object_detection.models.backbone.projector import MultiScaleProjector

__all__ = ["Backbone"]


@keras.saving.register_keras_serializable(package="backbone")
class Backbone(layers.Layer):
    """Keras3 port of the RF-DETR Backbone.

    Wraps a DinoV2 encoder and a MultiScaleProjector.
    Parses the `name` string to determine model variant
    (e.g. "dinov2_registers_windowed_base").
    """

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,
        drop_path=0.0,
        out_channels=256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov2_weights: bool = True,
        patch_size: int = 14,
        num_windows: int = 4,
        positional_encoding_size: int = 37,
        **kwargs,
    ):
        # Store the backbone name before calling super().__init__
        self._backbone_name = name
        super().__init__(name="backbone", **kwargs)

        # Parse the name string
        name_parts = name.split("_")
        assert name_parts[0] == "dinov2"

        use_registers = False
        if "registers" in name_parts:
            use_registers = True
            name_parts.remove("registers")
        use_windowed_attn = False
        if "windowed" in name_parts:
            use_windowed_attn = True
            name_parts.remove("windowed")
        assert len(name_parts) == 2, (
            "name should be dinov2, then either registers, windowed, "
            "both, or none, then the size"
        )

        self.encoder = DinoV2(
            size=name_parts[-1],
            out_feature_indexes=out_feature_indexes,
            window_block_indexes=window_block_indexes,
            shape=target_shape,
            use_registers=use_registers,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
            name="encoder",
        )

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, (
            "only support projector scale P3/P4/P5/P6 in ascending order."
        )
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            input_scales=[1.0] * len(self.encoder._out_feature_channels), # Inputs are isotropic (P4 / Stride 16)
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            name="projector",
        )

        self._export = False

        # Store init config for serialization
        self._init_config = dict(
            name=self._backbone_name,
            pretrained_encoder=pretrained_encoder,
            window_block_indexes=window_block_indexes,
            drop_path=drop_path,
            out_channels=out_channels,
            out_feature_indexes=out_feature_indexes,
            projector_scale=projector_scale,
            use_cls_token=use_cls_token,
            freeze_encoder=freeze_encoder,
            layer_norm=layer_norm,
            target_shape=target_shape,
            rms_norm=rms_norm,
            backbone_lora=backbone_lora,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov2_weights=load_dinov2_weights,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
        )

    def call(self, images, mask=None, training=None):
        """Forward pass.

        Args:
            images: Input tensor (B, H, W, C) — channels-last.
            mask: Boolean mask (B, H, W).

        Returns:
            List of (feat, mask) tuples. Each feat is (B, H', W', C)
            in channels-last format (Keras convention).
            If mask is None, returned masks are all-False.
        """
        # DinoV2 expects (B, H, W, C), returns list of (B, h, w, embed_dim)
        feats = self.encoder(images, training=training)
        # Projector expects/returns (B, H, W, C) channels-last
        feats = self.projector(feats, training=training)

        out = []
        for feat in feats:
            feat_h = ops.shape(feat)[1]
            feat_w = ops.shape(feat)[2]
            if mask is not None:
                # Resize mask to feature spatial size via nearest interpolation
                m = ops.cast(mask, "float32")
                m = ops.expand_dims(m, axis=-1)  # (B, H, W, 1)
                m = ops.image.resize(
                    m, (feat_h, feat_w),
                    interpolation="nearest",
                )
                m = ops.cast(ops.squeeze(m, axis=-1), "bool")  # (B, h, w)
            else:
                batch_size = ops.shape(feat)[0]
                m = ops.zeros((batch_size, feat_h, feat_w), dtype="bool")
            out.append((feat, m))
        return out

    def call_export(self, images, training=None):
        """Export-friendly forward (no NestedTensor).

        Returns:
            (feats, masks): lists of feature tensors and zero-masks.
        """
        feats = self.encoder(images, training=training)
        feats = self.projector(feats, training=training)
        out_feats = []
        out_masks = []
        for feat in feats:
            b = ops.shape(feat)[0]
            h = ops.shape(feat)[1]
            w = ops.shape(feat)[2]
            out_masks.append(ops.zeros((b, h, w), dtype="bool"))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_config(self):
        config = super().get_config()
        config.update(self._init_config)
        return config


# ─── Utility functions ───────────────────────────────────────────────────


def get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(name[name.find(".layer."):].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    """Determine weight decay rate based on parameter name."""
    if (
        ("gamma" in name)
        or ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("embeddings" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate