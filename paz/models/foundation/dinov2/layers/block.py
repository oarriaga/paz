from keras import layers

from .attention import attend
from .drop_path import apply_drop_path
from .layer_scale import apply_layer_scale
from .mlp import mlp
from .swiglu_ffn import swiglu_ffn_fused


def block(x, dim, num_heads, use_QKV_bias, use_projection_bias, attention_dropout_rate, FFN_layer, MLP_ratio, use_FFN_bias, activation, drop_rate, init_values, drop_path, normalization_layer, name):  # fmt: skip
    pre_attention = normalization_layer(name=f"{name}_norm1")(x)
    attended = apply_attention(pre_attention, dim, num_heads, use_QKV_bias, use_projection_bias, attention_dropout_rate, drop_rate, name)  # fmt: skip
    x = merge_residual(x, attended, dim, init_values, drop_path, name, 1)
    pre_FFN = normalization_layer(name=f"{name}_norm2")(x)
    forwarded = apply_FFN(pre_FFN, dim, FFN_layer, MLP_ratio, use_FFN_bias, activation, drop_rate, f"{name}_mlp")  # fmt: skip
    x = merge_residual(x, forwarded, dim, init_values, drop_path, name, 2)
    return x


def apply_attention(x, dim, num_heads, use_QKV_bias, use_projection_bias, attention_dropout_rate, drop_rate, name):  # fmt: skip
    head_dim = dim // num_heads
    rates = (attention_dropout_rate, drop_rate)
    return attend(x, num_heads, head_dim, *rates, use_QKV_bias, use_projection_bias, name)  # fmt: skip


def apply_FFN(x, dim, FFN_layer, MLP_ratio, use_FFN_bias, activation, drop_rate, name):  # fmt: skip
    hidden_dim = int(dim * MLP_ratio)
    FFN_inputs = (x, hidden_dim, dim, use_FFN_bias, drop_rate, name, activation)
    if FFN_layer == "mlp":
        output = mlp(*FFN_inputs)
    else:
        output = swiglu_ffn_fused(*FFN_inputs)
    return output


def merge_residual(x, branch, dim, init_values, drop_path, name, index):
    scaled = apply_layer_scale(branch, dim, init_values, f"{name}_ls{index}")
    dropped = apply_drop_path(scaled, drop_path, f"{name}_drop_path{index}")
    return layers.Add(name=f"{name}_add{index}")([x, dropped])
