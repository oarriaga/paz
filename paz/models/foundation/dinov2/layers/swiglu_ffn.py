import types
import keras
from keras import ops


def compute_swiglu_hidden(activation_layer, gate_and_value):
    value, gate = ops.split(gate_and_value, 2, axis=-1)
    return activation_layer(value) * gate


def build_dense(units, use_bias, name):
    return keras.layers.Dense(units=units, use_bias=use_bias, name=name)


def build_silu_activation():
    return keras.layers.Activation("silu")


def build_swiglu_ffn_layers(hid_dim, out_dim, use_bias):
    args = 2 * hid_dim, use_bias, "fused_gate_and_value_projection"
    fused_proj = build_dense(*args)
    out_proj = build_dense(out_dim, use_bias, "output_projection")
    return fused_proj, out_proj


def build_swiglu_aligned_layers(hid_dim, out_dim, use_bias):
    val_proj = build_dense(hid_dim, use_bias, "value_projection")
    gate_proj = build_dense(hid_dim, use_bias, "gate_projection")
    out_proj = build_dense(out_dim, use_bias, "output_projection")
    return val_proj, gate_proj, out_proj


def compute_effective_dims_standard(dim_in, dim_hid, dim_out):
    out_dim = dim_out if dim_out is not None else dim_in
    hid_dim = dim_hid if dim_hid is not None else dim_in
    return hid_dim, out_dim


def compute_effective_dims_fused(dim_in, dim_hid, dim_out):
    out_dim = dim_out if dim_out is not None else dim_in
    hid_raw = dim_hid if dim_hid is not None else dim_in
    hid_dim = (int(hid_raw * 2 / 3) + 7) // 8 * 8
    return hid_dim, out_dim


def compute_effective_dims_aligned(dim_in, dim_hid, dim_out, align_to):
    out_dim = dim_out if dim_out is not None else dim_in
    hid_raw = dim_hid if dim_hid is not None else dim_in
    d = int(hid_raw * 2 / 3)
    hid_dim = d + (-d % align_to)
    return hid_dim, out_dim


def set_ffn_attributes(model, fused_proj, out_proj, activation):
    model.fused_gate_and_value_projection = fused_proj
    model.output_projection = out_proj
    model.activation_layer = activation


def set_swiglu_fused_attributes(model, fused_proj, out_proj, drop_rate):
    model.fused_gate_and_value_projection = fused_proj
    model.output_projection = out_proj
    model.activation_layer = build_silu_activation()
    model.drop_layer = keras.layers.Dropout(rate=drop_rate)


def set_aligned_attributes(model, val_proj, gate_proj, out_proj, activation):
    model.value_projection = val_proj
    model.gate_projection = gate_proj
    model.output_projection = out_proj
    model.activation_layer = activation


def apply_swiglu_ffn(self, x, training=None, **_):
    gate_and_value = self.fused_gate_and_value_projection(x)
    hidden = compute_swiglu_hidden(self.activation_layer, gate_and_value)
    return self.output_projection(hidden)


def apply_swiglu_ffn_fused(self, x, training=None, **_):
    gate_and_value = self.fused_gate_and_value_projection(x)
    hidden = compute_swiglu_hidden(self.activation_layer, gate_and_value)
    output = self.output_projection(hidden)
    return self.drop_layer(output, training=training)


def apply_swiglu_ffn_aligned(self, x, training=None, **_):
    value = self.value_projection(x)
    gate = self.gate_projection(x)
    hidden = self.activation_layer(value) * gate
    return self.output_projection(hidden)


def SwiGLUFFN(
    input_features,
    hidden_features=None,
    output_features=None,
    use_bias=True,
    activation_layer=None,
    drop_rate=0.0,
    **kwargs,
):
    args = input_features, hidden_features, output_features
    hid_dim, out_dim = compute_effective_dims_standard(*args)
    fused_proj, out_proj = build_swiglu_ffn_layers(hid_dim, out_dim, use_bias)
    activation = activation_layer or build_silu_activation()
    x_in = keras.Input(shape=(None, input_features))
    model = keras.Model(inputs=x_in, outputs=fused_proj(x_in), **kwargs)
    set_ffn_attributes(model, fused_proj, out_proj, activation)
    model.call = types.MethodType(apply_swiglu_ffn, model)
    return model


def SwiGLUFFNFused(
    input_features,
    hidden_features=None,
    output_features=None,
    use_bias=True,
    activation_layer=None,
    drop_rate=0.0,
    **kwargs,
):
    args = input_features, hidden_features, output_features
    hid_dim, out_dim = compute_effective_dims_fused(*args)
    fused_proj, out_proj = build_swiglu_ffn_layers(hid_dim, out_dim, use_bias)
    x_in = keras.Input(shape=(None, input_features))
    model = keras.Model(inputs=x_in, outputs=fused_proj(x_in), **kwargs)
    set_swiglu_fused_attributes(model, fused_proj, out_proj, drop_rate)
    model.call = types.MethodType(apply_swiglu_ffn_fused, model)
    return model


def SwiGLUFFNAligned(
    input_features,
    hidden_features=None,
    output_features=None,
    use_bias=True,
    align_to=8,
    activation_layer=None,
    drop_rate=0.0,
    **kwargs,
):
    args = input_features, hidden_features, output_features, align_to
    hid_dim, out_dim = compute_effective_dims_aligned(*args)
    layers_tuple = build_swiglu_aligned_layers(hid_dim, out_dim, use_bias)
    val_proj, gate_proj, out_proj = layers_tuple
    activation = activation_layer or build_silu_activation()
    x_in = keras.Input(shape=(None, input_features))
    model = keras.Model(inputs=x_in, outputs=val_proj(x_in), **kwargs)
    set_aligned_attributes(model, val_proj, gate_proj, out_proj, activation)
    model.call = types.MethodType(apply_swiglu_ffn_aligned, model)
    return model
