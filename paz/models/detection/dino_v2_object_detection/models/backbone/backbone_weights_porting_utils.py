import numpy as np
import torch

from paz.models.foundation.dinov2.models.windowed_vision_transformer import (
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


# ═══════════════════════════════════════════════════════════════════
# General helpers
# ═══════════════════════════════════════════════════════════════════

ATOL = 1e-5
RTOL = 1e-4


def to_keras(pt_tensor):
    """Convert a PyTorch tensor to a NumPy array.

    Args:
        pt_tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Detached NumPy array.
    """
    return pt_tensor.detach().cpu().numpy()


def assert_close(pt_tensor, keras_array, atol=ATOL, rtol=RTOL):
    """Assert numerical closeness between a PyTorch tensor and a NumPy array.

    Args:
        pt_tensor (torch.Tensor): Reference tensor.
        keras_array: Array to compare.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
    """
    pt_np = pt_tensor.detach().cpu().numpy()
    k_np = np.array(keras_array)
    np.testing.assert_allclose(k_np, pt_np, atol=atol, rtol=rtol)


def chw_to_hwc(x_np):
    """(B, C, H, W) → (B, H, W, C)"""
    return np.transpose(x_np, (0, 2, 3, 1))


def hwc_to_chw(x_np):
    """(B, H, W, C) → (B, C, H, W)"""
    return np.transpose(x_np, (0, 3, 1, 2))


# ═══════════════════════════════════════════════════════════════════
# Test data helpers
# ═══════════════════════════════════════════════════════════════════

def make_mask(batch, h, w, all_false=True):
    """Create a boolean mask.

    Args:
        batch (int): Batch size.
        h (int): Height.
        w (int): Width.
        all_false (bool): If True, return all-False mask.

    Returns:
        np.ndarray: Boolean mask of shape (batch, h, w).
    """
    if all_false:
        return np.zeros((batch, h, w), dtype=bool)
    rng = np.random.RandomState(42)
    return rng.rand(batch, h, w) > 0.5


def make_pt_nested_tensor(images_np, mask_np):
    """Create a NestedTensor-like object for PyTorch parity tests.

    Args:
        images_np (np.ndarray): Images in channels-last format (B, H, W, C).
        mask_np (np.ndarray): Boolean mask (B, H, W).

    Returns:
        NestedTensor: Object with .tensors (B, C, H, W) and .mask attributes.
    """
    class NestedTensor:
        def __init__(self, t, m):
            self.tensors = t
            self.mask = m
    images_chw = np.transpose(images_np, (0, 3, 1, 2))
    return NestedTensor(
        torch.from_numpy(images_chw),
        torch.from_numpy(mask_np),
    )


def build_keras_embed(keras_embed, batch_size, height, width, channels=3):
    """Build a Keras embedding layer by running dummy data through it.

    Args:
        keras_embed: Embedding layer to build.
        batch_size (int): Batch size.
        height (int): Image height.
        width (int): Image width.
        channels (int): Number of channels.
    """
    dummy = np.zeros((batch_size, height, width, channels), dtype=np.float32)
    keras_embed(dummy, training=False)


# ═══════════════════════════════════════════════════════════════════
# DinoV2 encoder weight transfer
# ═══════════════════════════════════════════════════════════════════

def transfer_conv2d(pt_conv, keras_conv):
    """Transfer Conv2D weights from PyTorch to Keras.

    Transposes kernel from (O, I, H, W) to (H, W, I, O).

    Args:
        pt_conv: PyTorch Conv2d module.
        keras_conv: Keras Conv2D layer.
    """
    w = to_keras(pt_conv.weight)
    w = np.transpose(w, (2, 3, 1, 0))
    b = to_keras(pt_conv.bias)
    keras_conv.set_weights([w, b])


def transfer_dense(pt_linear, keras_dense):
    """Transfer Dense layer weights.

    Transposes weight from (out, in) to (in, out).

    Args:
        pt_linear: PyTorch Linear module.
        keras_dense: Keras Dense layer.
    """
    w = to_keras(pt_linear.weight).T
    b = to_keras(pt_linear.bias)
    keras_dense.set_weights([w, b])


def transfer_layernorm(pt_ln, keras_ln):
    """Transfer LayerNorm weights.

    Args:
        pt_ln: PyTorch LayerNorm module.
        keras_ln: Keras LayerNormalization layer.
    """
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
    registers = _optional_table(keras_model, f"{prefix}_{REGISTER_TOKENS}")
    if pt_embed.register_tokens is not None and registers is not None:
        assign_table(registers, to_keras(pt_embed.register_tokens))


def _optional_table(keras_model, name):
    try:
        return keras_model.get_layer(name).embeddings
    except ValueError:
        return None


def assign_table(keras_table, pt_array):
    """Assign a (1, N, D) PyTorch token table into a (N, D) Keras Embedding."""
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
    proj = keras_model.get_layer(f"{layer_name}_{ATTENTION}_proj")
    transfer_dense(pt_attn.output.dense, proj)


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
    transfer_layernorm(pt_layer.norm1, keras_model.get_layer(f"{layer_name}_{NORM1}"))
    transfer_attention(pt_layer.attention, keras_model, layer_name)
    transfer_layer_scale(
        pt_layer.layer_scale1, keras_model, f"{layer_name}_{LAYER_SCALE1}"
    )
    transfer_layernorm(pt_layer.norm2, keras_model.get_layer(f"{layer_name}_{NORM2}"))
    if hasattr(pt_layer.mlp, "fc1"):
        transfer_mlp(pt_layer.mlp, keras_model, layer_name)
    else:
        transfer_swiglu(pt_layer.mlp, keras_model, layer_name)
    transfer_layer_scale(
        pt_layer.layer_scale2, keras_model, f"{layer_name}_{LAYER_SCALE2}"
    )


def transfer_encoder(pt_encoder, keras_model, prefix=ENCODER):
    for i, pt_l in enumerate(pt_encoder.layer):
        layer_name = f"{prefix}_{ENCODER_LAYER.format(i)}"
        if not has_layer(keras_model, f"{layer_name}_{NORM1}"):
            break
        transfer_layer(pt_l, keras_model, layer_name)


def has_layer(keras_model, name):
    try:
        keras_model.get_layer(name)
        return True
    except ValueError:
        return False


# ═══════════════════════════════════════════════════════════════════
# Projector weight transfer
# ═══════════════════════════════════════════════════════════════════

def copy_conv2d(torch_layer, keras_layer):
    """Copy Conv2D weights, transposing from (O, I, H, W) to (H, W, I, O).

    Args:
        torch_layer: PyTorch Conv2d module.
        keras_layer: Keras Conv2D layer.
    """
    w = torch_layer.weight.data.cpu().numpy()
    if keras_layer.use_bias and torch_layer.bias is not None:
        b = torch_layer.bias.data.cpu().numpy()
        keras_layer.set_weights([w.transpose(2, 3, 1, 0), b])
    else:
        keras_layer.set_weights([w.transpose(2, 3, 1, 0)])


def copy_bn(torch_layer, keras_layer):
    """Copy BatchNorm weights (gamma, beta, running_mean, running_var).

    Args:
        torch_layer: PyTorch BatchNorm module.
        keras_layer: Keras BatchNormalization layer.
    """
    w = torch_layer.weight.data.cpu().numpy()
    b = torch_layer.bias.data.cpu().numpy()
    rm = torch_layer.running_mean.data.cpu().numpy()
    rv = torch_layer.running_var.data.cpu().numpy()
    keras_layer.set_weights([w, b, rm, rv])


def copy_ln(torch_layer, keras_layer):
    """Copy LayerNorm weights.

    Args:
        torch_layer: PyTorch LayerNorm module.
        keras_layer: Keras LayerNormalization layer.
    """
    w = torch_layer.weight.data.cpu().numpy()
    b = torch_layer.bias.data.cpu().numpy()
    keras_layer.set_weights([w, b])


def copy_weights_convx(torch_module, keras_module):
    """Copy ConvX block weights (convolution + normalization).

    Args:
        torch_module: PyTorch ConvX module.
        keras_module: Keras ConvX layer.
    """
    copy_conv2d(torch_module.conv, keras_module.conv)
    if hasattr(torch_module, "bn"):
        if isinstance(torch_module.bn, torch.nn.BatchNorm2d):
            copy_bn(torch_module.bn, keras_module.bn)
        elif isinstance(
            torch_module.bn,
            (torch.nn.LayerNorm, type(torch_module.bn)),
        ):
            copy_ln(torch_module.bn, keras_module.bn)


def copy_weights_c2f(torch_module, keras_module):
    """Copy C2f block weights (cv1, cv2, and bottleneck list).

    Args:
        torch_module: PyTorch C2f module.
        keras_module: Keras C2f layer.
    """
    copy_weights_convx(torch_module.cv1, keras_module.cv1)
    copy_weights_convx(torch_module.cv2, keras_module.cv2)
    for i, m_torch in enumerate(torch_module.m):
        m_keras = keras_module.m[i]
        copy_weights_convx(m_torch.cv1, m_keras.cv1)
        copy_weights_convx(m_torch.cv2, m_keras.cv2)


def port_weights_multiscale_projector(torch_model, keras_model):
    """Transfer all weights from a MultiScaleProjector.

    Copies sampling blocks (Conv2DTranspose, LayerNorm, ConvX)
    and stage blocks (C2f + normalization).

    Args:
        torch_model: PyTorch MultiScaleProjector.
        keras_model: Keras MultiScaleProjector.
    """
    from keras import layers

    for i in range(len(torch_model.stages_sampling)):
        for j in range(len(torch_model.stages_sampling[i])):
            t_sub = torch_model.stages_sampling[i][j]
            k_sub = keras_model.stages_sampling_blocks[i][j]

            if isinstance(k_sub, layers.Identity):
                continue

            k_idx = 0
            for t_layer in t_sub:
                if isinstance(t_layer, torch.nn.ConvTranspose2d):
                    k_layer = k_sub.layers[k_idx]
                    w = t_layer.weight.data.cpu().numpy()
                    if k_layer.use_bias and t_layer.bias is not None:
                        b = t_layer.bias.data.cpu().numpy()
                        k_layer.set_weights([w.transpose(2, 3, 1, 0), b])
                    else:
                        k_layer.set_weights([w.transpose(2, 3, 1, 0)])
                    k_idx += 1
                elif isinstance(t_layer, torch.nn.GELU):
                    k_idx += 1
                    continue
                elif hasattr(t_layer, "weight") and hasattr(
                    t_layer, "normalized_shape"
                ):
                    k_layer = k_sub.layers[k_idx]
                    copy_ln(t_layer, k_layer)
                    k_idx += 1
                elif hasattr(t_layer, "conv") and hasattr(t_layer, "bn"):
                    k_layer = k_sub.layers[k_idx]
                    copy_weights_convx(t_layer, k_layer)
                    k_idx += 1
                elif isinstance(t_layer, torch.nn.Conv2d):
                    k_layer = k_sub.layers[k_idx]
                    copy_conv2d(t_layer, k_layer)
                    k_idx += 1

    # Copy stage blocks (C2f + Norm)
    for i in range(len(torch_model.stages)):
        t_seq = torch_model.stages[i]
        k_seq = keras_model.stages_blocks[i]
        copy_weights_c2f(t_seq[0], k_seq.layers[0])
        copy_ln(t_seq[1], k_seq.layers[1])
