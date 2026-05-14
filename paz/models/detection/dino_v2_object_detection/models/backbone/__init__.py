from .dinov2_with_windowed_attn import (
    WindowedDinov2PatchEmbeddings,
    WindowedDinov2Layer,
    WindowedDinov2Encoder,
    WindowedDinov2Model,
    dinov2_windowed_small,
    dinov2_windowed_base,
    dinov2_windowed_large,
    dinov2_windowed_giant,
)
from .dinov2 import DinoV2
from .projector import MultiScaleProjector
from .backbone import Backbone, get_dinov2_lr_decay_rate, get_dinov2_weight_decay_rate
from .position_encoding import PositionEmbeddingSine, build_position_encoding

import keras


@__import__("keras").saving.register_keras_serializable(package="backbone")
class Joiner(__import__("keras").layers.Layer):
    """Combines a Backbone with sinusoidal position embeddings.

    Attributes:
        backbone (Backbone): Feature extraction backbone.
        position_embedding (PositionEmbeddingSine): Positional encoding layer.
    """

    def __init__(self, backbone, position_embedding, **kwargs):
        super().__init__(name="joiner", **kwargs)
        self.backbone = backbone
        self.position_embedding = position_embedding
        self._export = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "position_embedding": keras.saving.serialize_keras_object(
                    self.position_embedding
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["backbone"] = keras.saving.deserialize_keras_object(config["backbone"])
        config["position_embedding"] = keras.saving.deserialize_keras_object(
            config["position_embedding"]
        )
        return cls(**config)

    def call(self, images, mask=None, training=None):
        """Run backbone and compute position encodings per scale.

        Args:
            images (Tensor): Input tensor of shape (B, H, W, C).
            mask (Tensor): Boolean mask of shape (B, H, W).
            training (bool): Whether in training mode.

        Returns:
            tuple: (features, positions) where features is a list of
                (feat, mask) tuples and positions is a list of
                position encoding tensors.
        """
        x = self.backbone(images, mask=mask, training=training)
        pos = []
        for feat, feat_mask in x:
            p = self.position_embedding(
                feat_mask,
                align_dim_orders=False,
            )
            pos.append(p)
        return x, pos

    def call_export(self, images, training=None):
        """Export-friendly forward pass without nested tensor tuples.

        Args:
            images (Tensor): Input tensor of shape (B, H, W, C).
            training (bool): Whether in training mode.

        Returns:
            tuple: (feats, None, positions) where feats and positions
                are flat lists of tensors.
        """
        feats, masks = self.backbone.call_export(images, training=training)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self.position_embedding(mask, align_dim_orders=False))
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers=None,
    pretrained_encoder=None,
    window_block_indexes=None,
    drop_path=0.0,
    out_channels=256,
    out_feature_indexes=None,
    projector_scale=None,
    use_cls_token=False,
    hidden_dim=256,
    position_embedding="sine",
    freeze_encoder=False,
    layer_norm=False,
    target_shape=(640, 640),
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    gradient_checkpointing=False,
    load_dinov2_weights=True,
    patch_size=14,
    num_windows=4,
    positional_encoding_size=37,
):
    """Build a Joiner by assembling a Backbone with sinusoidal position encoding.

    Args:
        encoder (str): Backbone name (e.g. 'dinov2_windowed_small').
        hidden_dim (int): Hidden dimension for position encoding.
        position_embedding (str): Type of position embedding ('sine' or 'v2').

    Returns:
        Joiner: Combined backbone and position encoding module.
    """
    pos_embed = build_position_encoding(hidden_dim, position_embedding)

    backbone = Backbone(
        name=encoder,
        pretrained_encoder=pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
    )

    model = Joiner(backbone, pos_embed)
    return model
