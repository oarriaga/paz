from collections import namedtuple
from keras import layers

from .attention import attend
from .drop_path import apply_drop_path
from .layer_scale import apply_layer_scale
from .mlp import mlp
from .swiglu_ffn import swiglu_ffn_fused

AttentionArgs = namedtuple(
    "AttentionArgs", "num_heads use_qkv_bias use_proj_bias attn_drop_rate"
)
FFNArgs = namedtuple("FFNArgs", "FFN_layer mlp_ratio use_FFN_bias activation")


def block(
    x, dim, attention, FFN, drop_rate, init_values, drop_path, normalization_layer, name
):
    pre_attention = normalization_layer(name=f"{name}_norm1")(x)
    attended = apply_attention(pre_attention, dim, attention, drop_rate, name)
    x = merge_residual(x, attended, dim, init_values, drop_path, name, 1)
    pre_FFN = normalization_layer(name=f"{name}_norm2")(x)
    forwarded = apply_FFN(pre_FFN, dim, FFN, drop_rate, f"{name}_mlp")
    x = merge_residual(x, forwarded, dim, init_values, drop_path, name, 2)
    return x


def apply_attention(x, dim, attention, drop_rate, name):
    head_dim = dim // attention.num_heads
    attn_args = (
        attention.num_heads,
        head_dim,
        attention.attn_drop_rate,
        drop_rate,
        attention.use_qkv_bias,
        attention.use_proj_bias,
    )
    return attend(x, *attn_args, name)


def apply_FFN(x, dim, FFN, drop_rate, name):
    hidden = int(dim * FFN.mlp_ratio)
    args = (x, hidden, dim, FFN.use_FFN_bias, drop_rate, name, FFN.activation)
    if FFN.FFN_layer == "mlp":
        output = mlp(*args)
    else:
        output = swiglu_ffn_fused(*args)
    return output


def merge_residual(x, branch, dim, init_values, drop_path, name, index):
    scaled = apply_layer_scale(branch, dim, init_values, f"{name}_ls{index}")
    dropped = apply_drop_path(scaled, drop_path, f"{name}_drop_path{index}")
    return layers.Add(name=f"{name}_add{index}")([x, dropped])
