from keras import layers

from paz.layers import apply_drop_path, apply_layer_scale

from .attention import attend
from .mlp import mlp
from .swiglu_ffn import swiglu_ffn_fused


def block(x, dim, num_heads, use_qkv_bias, use_projection_bias, attention_dropout_rate, feedforward_layer, mlp_ratio, use_feedforward_bias, activation, drop_rate, init_values, drop_path, normalization_layer, name):  # fmt: skip
    pre_attention = normalization_layer(name=f"{name}_norm1")(x)
    attended = apply_attention(pre_attention, dim, num_heads, use_qkv_bias, use_projection_bias, attention_dropout_rate, drop_rate, name)  # fmt: skip
    x = merge_residual(x, attended, dim, init_values, drop_path, name, 1)
    pre_feedforward = normalization_layer(name=f"{name}_norm2")(x)
    forwarded = apply_feedforward(pre_feedforward, dim, feedforward_layer, mlp_ratio, use_feedforward_bias, activation, drop_rate, f"{name}_mlp")  # fmt: skip
    x = merge_residual(x, forwarded, dim, init_values, drop_path, name, 2)
    return x


def apply_attention(x, dim, num_heads, use_qkv_bias, use_projection_bias, attention_dropout_rate, drop_rate, name):  # fmt: skip
    head_dim = dim // num_heads
    rates = (attention_dropout_rate, drop_rate)
    biases = (use_qkv_bias, use_projection_bias)
    return attend(x, num_heads, head_dim, *rates, *biases, name)


def apply_feedforward(x, dim, feedforward_layer, mlp_ratio, use_feedforward_bias, activation, drop_rate, name):  # fmt: skip
    hidden_dim = int(dim * mlp_ratio)
    inputs = (x, hidden_dim, dim, use_feedforward_bias, drop_rate, name, activation)  # fmt: skip
    if feedforward_layer == "mlp":
        return mlp(*inputs)
    return swiglu_ffn_fused(*inputs)


def merge_residual(x, branch, dim, init_values, drop_path, name, index):
    scaled = apply_layer_scale(branch, dim, init_values, f"{name}_ls{index}")
    dropped = apply_drop_path(scaled, drop_path, f"{name}_drop_path{index}")
    return layers.Add(name=f"{name}_add{index}")([x, dropped])
