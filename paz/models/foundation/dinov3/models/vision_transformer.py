import numpy as np
import math
import os
from functools import partial
from typing import Any, Dict, List, Literal, Sequence, Union


from paz.models.foundation.dinov3.layers import (
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from paz.models.foundation.dinov3.utils import named_apply

import keras
from keras import layers, ops, random

ffn_layer_dict_keras = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}


# ==============================================================================
# Numerically Stable Keras Layer Implementation
# ==============================================================================
@keras.saving.register_keras_serializable()
class StableLayerNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma", shape=(dim,), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(dim,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        x_dtype = x.dtype
        x_f32 = ops.cast(x, "float32")
        mean = ops.mean(x_f32, axis=-1, keepdims=True)
        variance = ops.mean(ops.square(x_f32 - mean), axis=-1, keepdims=True)
        norm_x = (x_f32 - mean) * ops.rsqrt(variance + self.epsilon)
        norm_x = ops.cast(norm_x, x_dtype)
        return norm_x * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def init_weights_vit(module, name: str = ""):
    layer = module
    if isinstance(layer, layers.Dense):
        trunc_normal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
        layer.kernel.assign(
            trunc_normal(shape=layer.kernel.shape, dtype=layer.kernel.dtype)
        )
        if layer.use_bias and layer.bias is not None:
            layer.bias.assign(ops.zeros_like(layer.bias))
        if hasattr(layer, "bias_mask") and layer.bias_mask is not None:
            o = layer.units
            start_idx, end_idx = o // 3, 2 * o // 3
            new_mask_values = np.ones(layer.bias_mask.shape, dtype=np.float32)
            if start_idx < end_idx:
                new_mask_values[start_idx:end_idx] = 0
            layer.bias_mask.assign(new_mask_values)
    elif isinstance(layer, (layers.LayerNormalization, StableLayerNormalization)):
        if hasattr(layer, "gamma") and layer.gamma is not None:
            layer.gamma.assign(ops.ones_like(layer.gamma))
        if hasattr(layer, "beta") and layer.beta is not None:
            layer.beta.assign(ops.zeros_like(layer.beta))
    elif isinstance(layer, PatchEmbed):
        k = 1.0 / (layer.in_channels * (layer.patch_size[0] ** 2))
        limit = math.sqrt(k)
        initializer = keras.initializers.RandomUniform(minval=-limit, maxval=limit)
        proj_layer = layer.projection
        proj_layer.kernel.assign(
            initializer(shape=proj_layer.kernel.shape, dtype=proj_layer.kernel.dtype)
        )
        if proj_layer.use_bias and proj_layer.bias is not None:
            proj_layer.bias.assign(
                initializer(shape=proj_layer.bias.shape, dtype=proj_layer.bias.dtype)
            )
    elif hasattr(layer, "reset_parameters") and callable(layer.reset_parameters):
        layer.reset_parameters()


@keras.saving.register_keras_serializable(package="paz.dinov3")
class DinoVisionTransformer(keras.Model):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        layerscale_init: float | None = None,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"Keras model ignored kwargs: {ignored_kwargs}")

        self.img_size_config = img_size
        self.patch_size_config = patch_size
        self.in_chans_config = in_chans
        self.pos_embed_rope_base_config = pos_embed_rope_base
        self.pos_embed_rope_min_period_config = pos_embed_rope_min_period
        self.pos_embed_rope_max_period_config = pos_embed_rope_max_period
        self.pos_embed_rope_normalize_coords_config = pos_embed_rope_normalize_coords
        self.pos_embed_rope_shift_coords_config = pos_embed_rope_shift_coords
        self.pos_embed_rope_jitter_coords_config = pos_embed_rope_jitter_coords
        self.pos_embed_rope_rescale_coords_config = pos_embed_rope_rescale_coords
        self.pos_embed_rope_dtype_config = pos_embed_rope_dtype
        self.embed_dim_config = embed_dim
        self.depth_config = depth
        self.num_heads_config = num_heads
        self.ffn_ratio_config = ffn_ratio
        self.qkv_bias_config = qkv_bias
        self.proj_bias_config = proj_bias
        self.ffn_bias_config = ffn_bias
        self.drop_path_rate_config = drop_path_rate
        self.norm_layer_config = norm_layer
        self.ffn_layer_config = ffn_layer
        self.layerscale_init_config = layerscale_init
        self.n_storage_tokens_config = n_storage_tokens
        self.mask_k_bias_config = mask_k_bias
        self.untie_cls_and_patch_norms_config = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm_config = untie_global_and_local_cls_norm

        if norm_layer == "layernorm":
            norm_layer_cls = partial(StableLayerNormalization, epsilon=1e-6)

        elif norm_layer == "rmsnorm":
            norm_layer_cls = partial(RMSNorm, epsilon=1e-6)  # FIX: Use correct epsilon
        else:
            raise ValueError(f"Unknown norm_layer: {norm_layer}")

        ffn_layer_cls = ffn_layer_dict_keras[ffn_layer]

        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,
            flatten_embedding=False,
        )
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, embed_dim), initializer="zeros"
        )
        self.n_storage_tokens = n_storage_tokens
        self.storage_tokens = (
            self.add_weight(
                name="storage_tokens",
                shape=(1, n_storage_tokens, embed_dim),
                initializer="zeros",
            )
            if n_storage_tokens > 0
            else None
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype="float32",
        )
        self.blocks = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]
        self.norm = norm_layer_cls(name="norm")
        self.cls_norm = (
            norm_layer_cls(name="cls_norm") if untie_cls_and_patch_norms else None
        )
        self.local_cls_norm = (
            norm_layer_cls(name="local_cls_norm")
            if untie_global_and_local_cls_norm
            else None
        )
        self.head = layers.Identity(name="head")
        self.mask_token = self.add_weight(
            name="mask_token", shape=(1, embed_dim), initializer="random_normal"
        )
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm

    def init_weights(self):
        if hasattr(self.rope_embed, "_init_weights"):
            self.rope_embed._init_weights()
        self.cls_token.assign(random.normal(self.cls_token.shape, stddev=0.02))
        if self.storage_tokens is not None:
            self.storage_tokens.assign(
                random.normal(self.storage_tokens.shape, stddev=0.02)
            )
        self.mask_token.assign(ops.zeros_like(self.mask_token))
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x, masks=None):
        x_patched = self.patch_embed(x)
        B, H, W, C = ops.shape(x_patched)
        x_flat = ops.reshape(x_patched, (B, H * W, C))
        if masks is not None:
            mask_expanded = ops.expand_dims(masks, axis=-1)
            mask_token_b = ops.broadcast_to(self.mask_token, (B, H * W, C))
            x_flat = ops.where(
                mask_expanded, ops.cast(mask_token_b, x_flat.dtype), x_flat
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        storage_tokens = (
            self.storage_tokens
            if self.n_storage_tokens > 0
            else ops.zeros((1, 0, cls_token.shape[-1]), dtype=cls_token.dtype)
        )
        tokens = [
            ops.cast(ops.repeat(cls_token, B, axis=0), x_flat.dtype),
            ops.cast(ops.repeat(storage_tokens, B, axis=0), x_flat.dtype),
            x_flat,
        ]
        return ops.concatenate(tokens, axis=1), (H, W)

    def forward_features_list(
        self, x_list: List, masks_list: List, training: bool = False
    ) -> List[Dict[str, Any]]:
        x, rope = [], []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for blk in self.blocks:
            rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            x = blk(x, rope=rope_sincos, training=training)
        output = []
        for idx, (x_item, masks) in enumerate(zip(x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(
                        x_item[:, : self.n_storage_tokens + 1]
                    )
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(
                        x_item[:, : self.n_storage_tokens + 1]
                    )
                else:
                    x_norm_cls_reg = self.norm(x_item[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x_item[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x_item)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x_item,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, training=False):
        if isinstance(x, list):
            return self.forward_features_list(x, masks, training=training)
        else:
            return self.forward_features_list([x], [masks], training=training)[0]

    def _get_intermediate_layers_not_chunked(
        self, x, n: int = 1, training: bool = False
    ) -> List:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output = []
        total_block_len = len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        rope_sincos = self.rope_embed(H=H, W=W)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope=rope_sincos, training=training)
            if i in blocks_to_take:
                output.append(x)
        return output

    def get_intermediate_layers(
        self,
        x,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
        training: bool = False,
    ):
        outputs = self._get_intermediate_layers_not_chunked(x, n, training=training)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(
                        ops.concatenate([x_norm_cls_reg, x_norm_patch], axis=1)
                    )
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B = ops.shape(x)[0]
            # Keras input is channels-last, so H/W are at indices 1 and 2
            H_img, W_img = ops.shape(x)[1], ops.shape(x)[2]
            H, W = H_img // self.patch_size, W_img // self.patch_size
            outputs = [
                ops.transpose(ops.reshape(out, (B, H, W, -1)), [0, 3, 1, 2])
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        if return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def call(self, x, masks=None, training: bool = False):
        ret = self.forward_features(x, masks=masks, training=training)
        if training:
            return {
                "x_norm_clstoken": ret["x_norm_clstoken"],
                "x_norm_patchtokens": ret["x_norm_patchtokens"],
            }
        return self.head(ret["x_norm_clstoken"])

    def get_config(self):
        config = super().get_config()
        config.update({
            "img_size": self.img_size_config,
            "patch_size": self.patch_size_config,
            "in_chans": self.in_chans_config,
            "pos_embed_rope_base": self.pos_embed_rope_base_config,
            "pos_embed_rope_min_period": self.pos_embed_rope_min_period_config,
            "pos_embed_rope_max_period": self.pos_embed_rope_max_period_config,
            "pos_embed_rope_normalize_coords": self.pos_embed_rope_normalize_coords_config,
            "pos_embed_rope_shift_coords": self.pos_embed_rope_shift_coords_config,
            "pos_embed_rope_jitter_coords": self.pos_embed_rope_jitter_coords_config,
            "pos_embed_rope_rescale_coords": self.pos_embed_rope_rescale_coords_config,
            "pos_embed_rope_dtype": self.pos_embed_rope_dtype_config,
            "embed_dim": self.embed_dim_config,
            "depth": self.depth_config,
            "num_heads": self.num_heads_config,
            "ffn_ratio": self.ffn_ratio_config,
            "qkv_bias": self.qkv_bias_config,
            "proj_bias": self.proj_bias_config,
            "ffn_bias": self.ffn_bias_config,
            "drop_path_rate": self.drop_path_rate_config,
            "norm_layer": self.norm_layer_config,
            "ffn_layer": self.ffn_layer_config,
            "layerscale_init": self.layerscale_init_config,
            "n_storage_tokens": self.n_storage_tokens_config,
            "mask_k_bias": self.mask_k_bias_config,
            "untie_cls_and_patch_norms": self.untie_cls_and_patch_norms_config,
            "untie_global_and_local_cls_norm": self.untie_global_and_local_cls_norm_config,
        })
        return config


def load_pretrained_weights(model, model_name):
    weights_path = os.path.expanduser(f"~/.keras/paz/models/{model_name}.keras")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        # Build the model first with a dummy input
        img_size = model.img_size_config
        dummy_input = np.zeros((1, img_size, img_size, 3), dtype="float32")
        model(dummy_input, training=False)
        model.load_weights(weights_path)
    else:
        print(f"Weights file not found at {weights_path}. Model initialized with random weights.")


def vit_small(patch_size=16, weights=None, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "n_storage_tokens": 4,
        "layerscale_init": 1e-6,
        "pos_embed_rope_dtype": "float32",
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    if weights == "paz":
        load_pretrained_weights(model, "dinov3_vits16_ported")
    return model


def vit_base(patch_size=16, weights=None, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4,
        "n_storage_tokens": 4,
        "layerscale_init": 1e-6,
        "pos_embed_rope_dtype": "float32",
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    if weights == "paz":
        load_pretrained_weights(model, "dinov3_vitb16_ported")
    return model


def vit_large(patch_size=16, weights=None, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4,
        "n_storage_tokens": 4,
        "layerscale_init": 1e-6,
        "pos_embed_rope_dtype": "float32",
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    if weights == "paz":
        load_pretrained_weights(model, "dinov3_vitl16_ported")
    return model


def vit_so400m(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 1152,
        "depth": 27,
        "num_heads": 18,
        "ffn_ratio": 3.777777778,
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def vit_huge2(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 20,
        "ffn_ratio": 4,
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def vit_giant2(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    defaults = {
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_ratio": 4,
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def vit_7b(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 4096,
        "depth": 40,
        "num_heads": 32,
        "ffn_ratio": 3,
        "img_size": input_shape[0],
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def DINOV3VITS(input_shape=(224, 224, 3), **kwargs):
    return vit_small(weights="paz", input_shape=input_shape, **kwargs)


def DINOV3VITB(input_shape=(224, 224, 3), **kwargs):
    return vit_base(weights="paz", input_shape=input_shape, **kwargs)


def DINOV3VITL(input_shape=(224, 224, 3), **kwargs):
    return vit_large(weights="paz", input_shape=input_shape, **kwargs)
