from keras import layers, Model, ops
from paz.models.foundation.dinov2.models.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
    vit_giant2,
)
from examples.dino_object_detection.models.utils.misc import (
    NestedTensor,
)


import math
import keras
from keras import layers, ops, Model
from keras.saving import register_keras_serializable


@register_keras_serializable(package="MyLayers")
class PositionEmbeddingSine(layers.Layer):
    """
    Keras implementation of PositionEmbeddingSine identical to PyTorch DETR/DINO logic.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, mask):
        # Cast to float for cumsum
        not_mask = ops.cast(ops.logical_not(mask), "float32")

        # Cumulative sum along height (axis 1) and width (axis 2)
        y_embed = ops.cumsum(not_mask, axis=1)
        x_embed = ops.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            shape = ops.shape(y_embed)
            B, H, W = shape[0], shape[1], shape[2]
            y_last = ops.slice(y_embed, [0, H - 1, 0], [B, 1, W])
            x_last = ops.slice(x_embed, [0, 0, W - 1], [B, H, 1])

            y_embed = y_embed / (y_last + eps) * self.scale
            x_embed = x_embed / (x_last + eps) * self.scale

        # Create dim_t
        dim_t = ops.arange(self.num_pos_feats, dtype="float32")
        dim_t = self.temperature ** (2 * ops.floor(dim_t / 2) / self.num_pos_feats)

        # (B, H, W, 1) / (num_pos_feats) -> Broadcast
        pos_x = ops.expand_dims(x_embed, axis=-1) / dim_t
        pos_y = ops.expand_dims(y_embed, axis=-1) / dim_t

        # Interleave sine/cosine
        pos_x_sin = ops.sin(pos_x[..., 0::2])
        pos_x_cos = ops.cos(pos_x[..., 1::2])
        pos_x = ops.stack([pos_x_sin, pos_x_cos], axis=-1)

        # Flatten last two dims
        sx = ops.shape(pos_x)
        pos_x = ops.reshape(pos_x, (sx[0], sx[1], sx[2], -1))

        pos_y_sin = ops.sin(pos_y[..., 0::2])
        pos_y_cos = ops.cos(pos_y[..., 1::2])
        pos_y = ops.stack([pos_y_sin, pos_y_cos], axis=-1)
        sy = ops.shape(pos_y)
        pos_y = ops.reshape(pos_y, (sy[0], sy[1], sy[2], -1))

        # Concatenate Y and X embeddings
        pos = ops.concatenate([pos_y, pos_x], axis=-1)

        return pos


@register_keras_serializable(package="MyLayers")
class Joiner(Model):
    def __init__(self, backbone, position_embedding, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.position_embedding = position_embedding

    def call(self, inputs):
        # 1. Get Feature Maps from Backbone
        features = self.backbone(inputs)

        pos_embeds = []
        for x in features:
            # 2. Create Masks
            b_sz = ops.shape(x)[0]
            h_sz = ops.shape(x)[1]
            w_sz = ops.shape(x)[2]

            # Mask shape (B, H, W), all False (0) -> All Valid
            mask = ops.zeros((b_sz, h_sz, w_sz), dtype="bool")

            # 3. Compute Position Embeddings
            pos = self.position_embedding(mask)

            # Cast to match feature dtype (e.g. float16/float32)
            pos = ops.cast(pos, x.dtype)
            pos_embeds.append(pos)

        # Return tuple strictly: (features_list, pos_embeds_list)
        return features, pos_embeds


@register_keras_serializable(package="MyLayers")
class DinoV2BackboneWrapper(Model):
    def __init__(
        self,
        model_name,
        out_feature_indexes,
        patch_size=14,
        img_size=518,
        number_of_register_tokens=0,
        init_values=1e-5,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size

        model_map = {
            "dinov2_small": vit_small,
            "dinov2_base": vit_base,
            "dinov2_large": vit_large,
            "dinov2_giant": vit_giant2,
        }

        base_name = [k for k in model_map.keys() if k in model_name]
        if not base_name:
            raise ValueError(
                f"Unknown encoder: {model_name}. Available: {list(model_map.keys())}"
            )

        builder_fn = model_map[base_name[0]]

        # Instantiate the model with correct parameters
        self.feature_extractor = builder_fn(
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            number_of_register_tokens=number_of_register_tokens,
            drop_path_rate=drop_path_rate,
            name="dinov2_encoder",
        )

        embed_dim = self.feature_extractor.embedding_dimension
        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)

    def call(self, inputs, training=None):
        # 1. Prepare tokens (Embeddings + Positional Encoding + CLS + Registers)
        x = self.feature_extractor.prepare_tokens_with_masks(inputs)

        # 2. Iterate Blocks
        outputs = []

        # Handle chunked blocks if present
        if (
            hasattr(self.feature_extractor, "chunked_blocks")
            and self.feature_extractor.chunked_blocks
        ):
            all_blocks = []
            for chunk in self.feature_extractor.blocks:
                all_blocks.extend(chunk.blocks)
        else:
            all_blocks = self.feature_extractor.blocks

        norm_layer = self.feature_extractor.normalization

        for i, block in enumerate(all_blocks):
            # Pass training flag for DropPath/Dropout
            x = block(x, training=training)

            if i in self.out_feature_indexes:
                outputs.append(norm_layer(x))

        # 3. Remove CLS/Register tokens
        start_index = 1 + self.feature_extractor.number_of_register_tokens
        outputs = [out[:, start_index:] for out in outputs]

        # 4. Reshape to (B, H, W, C)
        B = ops.shape(inputs)[0]
        H = ops.shape(inputs)[1]
        W = ops.shape(inputs)[2]
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        embed_dim = self.feature_extractor.embedding_dimension
        outputs = [
            ops.reshape(out, (B, patch_h, patch_w, embed_dim)) for out in outputs
        ]

        return outputs

    @property
    def model(self):
        return self.feature_extractor

    def build(self, input_shape):
        if hasattr(self, "feature_extractor"):
            if isinstance(self.feature_extractor, Model):
                if not self.feature_extractor.built:
                    self.feature_extractor.build(input_shape)

        embed_dim = self.feature_extractor.embedding_dimension
        self._out_feature_channels = [embed_dim] * len(self.out_feature_indexes)

        super().build(input_shape)


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov2_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
    num_register_tokens=0,
    init_values=1e-5,
):
    # Adjust out_feature_indexes for Keras 0-based indexing
    keras_out_feature_indexes = [i - 1 for i in out_feature_indexes if i > 0]

    backbone = DinoV2BackboneWrapper(
        model_name=encoder,
        out_feature_indexes=keras_out_feature_indexes,
        patch_size=patch_size,
        img_size=target_shape[0],
        number_of_register_tokens=num_register_tokens,
        init_values=init_values,
        drop_path_rate=drop_path,
    )

    if freeze_encoder:
        backbone.trainable = False

    if projector_scale:
        from examples.dino_object_detection.models.backbone.projector import (
            MultiScaleProjector,
        )

        level2scalefactor = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}
        scale_factors = [level2scalefactor[lvl] for lvl in projector_scale]

        projector = MultiScaleProjector(
            in_channels=backbone._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
    else:
        projector = None

    inputs = layers.Input(shape=(target_shape[0], target_shape[1], 3))

    # Features are (B, H, W, C)
    features = backbone(inputs)

    if projector:

        features = [ops.transpose(f, (0, 3, 1, 2)) for f in features]
        features = projector(features)
        features = [ops.transpose(f, (0, 2, 3, 1)) for f in features]

    model = Model(inputs=inputs, outputs=features, name="backbone_joiner")

    position_embedding_layer = PositionEmbeddingSine(
        num_pos_feats=hidden_dim // 2, normalize=True
    )

    joiner_model = Joiner(
        backbone=model,
        position_embedding=position_embedding_layer,
        name="joiner",
    )

    return joiner_model
