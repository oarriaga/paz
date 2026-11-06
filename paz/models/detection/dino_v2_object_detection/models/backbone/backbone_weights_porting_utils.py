from collections import namedtuple

import numpy as np
import torch

from paz.models.foundation.dinov2_legacy.models.windowed_vision_transformer import (
    EMBEDDINGS,
    ENCODER,
    ENCODER_LAYER,
    NORM1,
    NORM2,
    ATTENTION,
    LAYER_SCALE1,
    LAYER_SCALE2,
    FFN,
    PATCH_PROJECTION,
    CLS_TOKEN,
    POS_EMBED,
    REGISTER_TOKENS,
)


ATOL = 1e-5
RTOL = 1e-4

NestedTensor = namedtuple("NestedTensor", "tensors mask")


def to_keras(pt_tensor):
    return pt_tensor.detach().cpu().numpy()


def assert_close(pt_tensor, keras_array, atol=ATOL, rtol=RTOL):
    pt_np = pt_tensor.detach().cpu().numpy()
    k_np = np.array(keras_array)
    np.testing.assert_allclose(k_np, pt_np, atol=atol, rtol=rtol)


def chw_to_hwc(x_np):
    return np.transpose(x_np, (0, 2, 3, 1))


def hwc_to_chw(x_np):
    return np.transpose(x_np, (0, 3, 1, 2))


def make_mask(batch, h, w, all_false=True):
    result = np.zeros((batch, h, w), dtype=bool)
    if not all_false:
        rng = np.random.RandomState(42)
        result = rng.rand(batch, h, w) > 0.5
    return result


def make_pt_nested_tensor(images_np, mask_np):
    images_chw = np.transpose(images_np, (0, 3, 1, 2))
    tensors = torch.from_numpy(images_chw)
    mask = torch.from_numpy(mask_np)
    return NestedTensor(tensors, mask)


def build_keras_embed(keras_embed, batch_size, height, width, channels=3):
    dummy = np.zeros((batch_size, height, width, channels), dtype=np.float32)
    keras_embed(dummy, training=False)


def transfer_conv2d(pt_conv, keras_conv):
    w = to_keras(pt_conv.weight)
    w = np.transpose(w, (2, 3, 1, 0))  # PT (O,I,H,W) -> Keras (H,W,I,O)
    b = to_keras(pt_conv.bias)
    keras_conv.set_weights([w, b])


def transfer_dense(pt_linear, keras_dense):
    w = to_keras(pt_linear.weight).T  # PT (out,in) -> Keras (in,out)
    b = to_keras(pt_linear.bias)
    keras_dense.set_weights([w, b])


def transfer_layernorm(pt_ln, keras_ln):
    keras_ln.set_weights([to_keras(pt_ln.weight), to_keras(pt_ln.bias)])


def transfer_layer_scale(pt_ls, keras_model, name):
    keras_model.get_layer(name).kernel.assign(to_keras(pt_ls.lambda1))


def transfer_patch_embeddings(pt_embed, keras_model, prefix=EMBEDDINGS):
    conv = keras_model.get_layer(f"{prefix}_{PATCH_PROJECTION}")
    transfer_conv2d(pt_embed.patch_embeddings.projection, conv)
    cls = keras_model.get_layer(f"{prefix}_{CLS_TOKEN}").embeddings
    assign_table(cls, to_keras(pt_embed.cls_token))
    pos = keras_model.get_layer(f"{prefix}_{POS_EMBED}").embeddings
    assign_table(pos, to_keras(pt_embed.position_embeddings))
    name = f"{prefix}_{REGISTER_TOKENS}"
    registers = optional_embedding_table(keras_model, name)
    if pt_embed.register_tokens is not None and registers is not None:
        assign_table(registers, to_keras(pt_embed.register_tokens))


def optional_embedding_table(keras_model, name):
    result = None
    try:
        result = keras_model.get_layer(name).embeddings
    except ValueError:
        pass
    return result


def assign_table(keras_table, pt_array):
    # PT (1,N,D) token table -> Keras (N,D) Embedding
    keras_table.assign(np.reshape(pt_array, keras_table.shape))


def transfer_attention(pt_attn, keras_model, layer_name):
    q_w = to_keras(pt_attn.attention.query.weight).T
    k_w = to_keras(pt_attn.attention.key.weight).T
    v_w = to_keras(pt_attn.attention.value.weight).T
    q_b = to_keras(pt_attn.attention.query.bias)
    k_b = to_keras(pt_attn.attention.key.bias)
    v_b = to_keras(pt_attn.attention.value.bias)
    fused_w = np.concatenate([q_w, k_w, v_w], axis=1)
    fused_b = np.concatenate([q_b, k_b, v_b], axis=0)
    qkv = keras_model.get_layer(f"{layer_name}_{ATTENTION}_qkv")
    qkv.set_weights([fused_w, fused_b])
    projection = keras_model.get_layer(f"{layer_name}_{ATTENTION}_proj")
    transfer_dense(pt_attn.output.dense, projection)


def transfer_mlp(pt_mlp, keras_model, layer_name):
    ffn = f"{layer_name}_{FFN}"
    transfer_dense(pt_mlp.fc1, keras_model.get_layer(f"{ffn}_fc1"))
    transfer_dense(pt_mlp.fc2, keras_model.get_layer(f"{ffn}_fc2"))


def transfer_swiglu(pt_swiglu, keras_model, layer_name):
    ffn = f"{layer_name}_{FFN}"
    gate = keras_model.get_layer(f"{ffn}_fused_gate_and_value_projection")
    transfer_dense(pt_swiglu.weights_in, gate)
    output = keras_model.get_layer(f"{ffn}_output_projection")
    transfer_dense(pt_swiglu.weights_out, output)


def transfer_layer(pt_layer, keras_model, layer_name):
    norm1 = keras_model.get_layer(f"{layer_name}_{NORM1}")
    transfer_layernorm(pt_layer.norm1, norm1)
    transfer_attention(pt_layer.attention, keras_model, layer_name)
    transfer_layer_scale(
        pt_layer.layer_scale1, keras_model, f"{layer_name}_{LAYER_SCALE1}"
    )
    norm2 = keras_model.get_layer(f"{layer_name}_{NORM2}")
    transfer_layernorm(pt_layer.norm2, norm2)
    if hasattr(pt_layer.mlp, "fc1"):
        transfer_mlp(pt_layer.mlp, keras_model, layer_name)
    else:
        transfer_swiglu(pt_layer.mlp, keras_model, layer_name)
    transfer_layer_scale(
        pt_layer.layer_scale2, keras_model, f"{layer_name}_{LAYER_SCALE2}"
    )


def transfer_encoder(pt_encoder, keras_model, prefix=ENCODER):
    for index, torch_layer in enumerate(pt_encoder.layer):
        layer_name = f"{prefix}_{ENCODER_LAYER.format(index)}"
        if not has_layer(keras_model, f"{layer_name}_{NORM1}"):
            break
        transfer_layer(torch_layer, keras_model, layer_name)


def has_layer(keras_model, name):
    result = True
    try:
        keras_model.get_layer(name)
    except ValueError:
        result = False
    return result


def copy_conv2d(torch_layer, keras_layer):
    # PT (O,I,H,W) -> Keras (H,W,I,O)
    w = torch_layer.weight.data.cpu().numpy()
    if keras_layer.use_bias and torch_layer.bias is not None:
        b = torch_layer.bias.data.cpu().numpy()
        keras_layer.set_weights([w.transpose(2, 3, 1, 0), b])
    else:
        keras_layer.set_weights([w.transpose(2, 3, 1, 0)])


def copy_bn(torch_layer, keras_layer):
    w = torch_layer.weight.data.cpu().numpy()
    b = torch_layer.bias.data.cpu().numpy()
    rm = torch_layer.running_mean.data.cpu().numpy()
    rv = torch_layer.running_var.data.cpu().numpy()
    keras_layer.set_weights([w, b, rm, rv])


def copy_ln(torch_layer, keras_layer):
    w = torch_layer.weight.data.cpu().numpy()
    b = torch_layer.bias.data.cpu().numpy()
    keras_layer.set_weights([w, b])


def copy_conv_transpose(torch_layer, keras_layer):
    # PT ConvTranspose (I,O,H,W) -> Keras (H,W,O,I)
    w = torch_layer.weight.data.cpu().numpy().transpose(2, 3, 1, 0)
    if keras_layer.use_bias and torch_layer.bias is not None:
        b = torch_layer.bias.data.cpu().numpy()
        keras_layer.set_weights([w, b])
    else:
        keras_layer.set_weights([w])


def copy_normalization(torch_norm, keras_norm):
    if isinstance(torch_norm, torch.nn.BatchNorm2d):
        copy_bn(torch_norm, keras_norm)
    else:
        copy_ln(torch_norm, keras_norm)


def copy_weights_convx(torch_convx, keras_model, name):
    copy_conv2d(torch_convx.conv, keras_model.get_layer(f"{name}_conv"))
    if hasattr(torch_convx, "bn"):
        keras_norm = keras_model.get_layer(f"{name}_bn")
        copy_normalization(torch_convx.bn, keras_norm)


def copy_weights_c2f(torch_c2f, keras_model, name):
    copy_weights_convx(torch_c2f.cv1, keras_model, f"{name}_cv1")
    copy_weights_convx(torch_c2f.cv2, keras_model, f"{name}_cv2")
    for index, bottleneck in enumerate(torch_c2f.m):
        copy_weights_convx(bottleneck.cv1, keras_model, f"{name}_m_{index}_cv1")
        copy_weights_convx(bottleneck.cv2, keras_model, f"{name}_m_{index}_cv2")


def is_torch_layernorm(torch_layer):
    return hasattr(torch_layer, "weight") and hasattr(torch_layer, "normalized_shape")  # fmt: skip


def copy_sampler_side_layer(torch_layer, keras_model, name):
    # GELU carries no weights, so it falls through every branch untouched.
    is_convx = hasattr(torch_layer, "conv") and hasattr(torch_layer, "bn")
    if is_torch_layernorm(torch_layer):
        copy_ln(torch_layer, keras_model.get_layer(f"{name}_ctx1_norm"))
    elif is_convx:
        copy_weights_convx(torch_layer, keras_model, f"{name}_cvx")
    elif isinstance(torch_layer, torch.nn.Conv2d):
        copy_conv2d(torch_layer, keras_model.get_layer(f"{name}_conv"))


def copy_sampler_layer(torch_layer, keras_model, name, transpose_index):
    # transpose_index drives the `_ctx{index}` layer names, so it must be
    # incremented in the same iteration order the builder used.
    if isinstance(torch_layer, torch.nn.ConvTranspose2d):
        keras_layer = keras_model.get_layer(f"{name}_ctx{transpose_index}")
        copy_conv_transpose(torch_layer, keras_layer)
        transpose_index = transpose_index + 1
    else:
        copy_sampler_side_layer(torch_layer, keras_model, name)
    return transpose_index


def copy_weights_sampler(torch_sampler, keras_model, name):
    transpose_index = 1
    for torch_layer in torch_sampler:
        args = (torch_layer, keras_model, name)
        transpose_index = copy_sampler_layer(*args, transpose_index)


def copy_projector_samplers(torch_projector, keras_model):
    for stage, samplers in enumerate(torch_projector.stages_sampling):
        for index, torch_sampler in enumerate(samplers):
            name = f"stage_{stage}_samp_{index}"
            copy_weights_sampler(torch_sampler, keras_model, name)


def port_weights_multiscale_projector(torch_projector, keras_model):
    copy_projector_samplers(torch_projector, keras_model)
    for stage, torch_stage in enumerate(torch_projector.stages):
        copy_weights_c2f(torch_stage[0], keras_model, f"stage_{stage}_c2f")
        copy_ln(torch_stage[1], keras_model.get_layer(f"stage_{stage}_norm"))
