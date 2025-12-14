import keras
from keras import layers, ops
from typing import Callable, Optional

from paz.models.foundation.dinov3.layers.attention import (
    SelfAttention,
    CausalSelfAttention,
)
from paz.models.foundation.dinov3.layers.ffn_layers import Mlp
from paz.models.foundation.dinov3.layers.layer_scale import LayerScale


def apply_stochastic_depth(x, drop_rate, training):
    if drop_rate == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_rate
    shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
    random_tensor = keras.random.uniform(shape, 0, 1, dtype=x.dtype)
    keep_mask = random_tensor >= drop_rate
    return (x / keep_prob) * ops.cast(keep_mask, x.dtype)


def apply_attention_with_residual(x, norm_layer, attn_layer, layer_scale, drop_path, rope, training):
    x_normalized = norm_layer(x)
    attention_output = attn_layer(x_normalized, rope=rope, training=training)
    scaled_output = layer_scale(attention_output, training=training)
    dropped_output = drop_path(scaled_output, training=training)
    return x + dropped_output


def apply_mlp_with_residual(x, norm_layer, mlp_layer, layer_scale, drop_path, training):
    x_normalized = norm_layer(x)
    mlp_output = mlp_layer(x_normalized, training=training)
    scaled_output = layer_scale(mlp_output, training=training)
    dropped_output = drop_path(scaled_output, training=training)
    return x + dropped_output


def process_attention_block_tensor(x, norm1, attn, ls1, drop_path1, norm2, mlp, ls2, drop_path2, rope, training):
    x = apply_attention_with_residual(x, norm1, attn, ls1, drop_path1, rope, training)
    x = apply_mlp_with_residual(x, norm2, mlp, ls2, drop_path2, training)
    return x


def process_attention_block_list(x_list, norm1, attn, ls1, drop_path1, norm2, mlp, ls2, drop_path2, rope_list, training):
    if rope_list is None:
        rope_list = [None] * len(x_list)

    x_norm1_list = [norm1(x) for x in x_list]
    attn_out_list = attn(x_norm1_list, rope=rope_list, training=training)
    x_res1_list = [
        x + drop_path1(ls1(attn_out), training=training)
        for x, attn_out in zip(x_list, attn_out_list)
    ]

    x_norm2_list = [norm2(x) for x in x_res1_list]
    mlp_out_list = mlp(x_norm2_list, training=training)
    x_res2_list = [
        x + drop_path2(ls2(mlp_out), training=training)
        for x, mlp_out in zip(x_res1_list, mlp_out_list)
    ]

    return x_res2_list


def process_causal_attention_block(x, attention_norm, attention, ls1, ffn_norm, feed_forward, ls2, is_causal, training):
    attn_input = attention_norm(x)
    attn_output = attention(attn_input, is_causal=is_causal, training=training)
    x = x + ls1(attn_output)

    ffn_input = ffn_norm(x)
    ffn_output = feed_forward(ffn_input, training=training)
    x = x + ls2(ffn_output)

    return x


class StochasticDepth(layers.Layer):
    def __init__(self, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        return apply_stochastic_depth(x, self.drop_rate, training)

    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


class _Identity(layers.Layer):
    def call(self, x, training=None):
        return x


class SelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer="gelu",
        norm_layer=layers.LayerNormalization,
        attn_class=SelfAttention,
        ffn_layer=Mlp,
        **kwargs,
    ):
        kwargs.pop("mask_k_bias", None)
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path_rate = drop_path
        self.init_values = init_values
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.attn_class = attn_class
        self.ffn_layer = ffn_layer

        self.norm1 = norm_layer(epsilon=1e-6, name="norm1")
        self.attn = attn_class(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name="attn",
        )
        self.ls1 = (
            LayerScale(dimension=dim, init_values=init_values, name="ls1")
            if init_values is not None
            else _Identity(name="ls1")
        )
        self.drop_path1 = StochasticDepth(drop_path) if drop_path > 0.0 else _Identity()

        self.norm2 = norm_layer(epsilon=1e-6, name="norm2")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            name="mlp",
        )
        self.ls2 = (
            LayerScale(dimension=dim, init_values=init_values, name="ls2")
            if init_values is not None
            else _Identity(name="ls2")
        )
        self.drop_path2 = StochasticDepth(drop_path) if drop_path > 0.0 else _Identity()

    def _forward_tensor(self, x, rope=None, training=None):
        return process_attention_block_tensor(
            x, self.norm1, self.attn, self.ls1, self.drop_path1,
            self.norm2, self.mlp, self.ls2, self.drop_path2,
            rope, training
        )

    def _forward_list(self, x_list, rope_list=None, training=None):
        return process_attention_block_list(
            x_list, self.norm1, self.attn, self.ls1, self.drop_path1,
            self.norm2, self.mlp, self.ls2, self.drop_path2,
            rope_list, training
        )

    def call(self, x, rope=None, training=None):
        if isinstance(x, list):
            return self._forward_list(x, rope_list=rope, training=training)
        else:
            return self._forward_tensor(x, rope=rope, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "proj_bias": self.proj_bias,
                "ffn_bias": self.ffn_bias,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path": self.drop_path_rate,
                "init_values": self.init_values,
                "act_layer": self.act_layer,
            }
        )
        return config


class CausalSelfAttentionBlock(layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = None,
        is_causal: bool = True,
        act_layer: str = "gelu",
        norm_layer: Callable = layers.LayerNormalization,
        drop: float = 0.0,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.init_values = init_values
        self.is_causal = is_causal
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.drop = drop
        self.qkv_bias = qkv_bias

        self.attention_norm = self.norm_layer(epsilon=1e-6, name="attention_norm")
        self.ffn_norm = self.norm_layer(epsilon=1e-6, name="ffn_norm")

        self.ls1 = (
            LayerScale(dimension=dim, init_values=init_values, name="ls1")
            if init_values is not None
            else _Identity(name="ls1")
        )
        self.ls2 = (
            LayerScale(dimension=dim, init_values=init_values, name="ls2")
            if init_values is not None
            else _Identity(name="ls2")
        )

        self.attention = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.drop,
            proj_drop=self.drop,
            name="attention",
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.feed_forward = Mlp(
            hidden_features=mlp_hidden_dim,
            out_features=self.dim,
            act_layer=self.act_layer,
            drop=self.drop,
            name="feed_forward",
        )

    def call(self, x, training=None):
        return process_causal_attention_block(
            x, self.attention_norm, self.attention, self.ls1,
            self.ffn_norm, self.feed_forward, self.ls2,
            self.is_causal, training
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "init_values": self.init_values,
                "is_causal": self.is_causal,
                "act_layer": self.act_layer,
                "drop": self.drop,
                "qkv_bias": self.qkv_bias,
            }
        )
        return config
