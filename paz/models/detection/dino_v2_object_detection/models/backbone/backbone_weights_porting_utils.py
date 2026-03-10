import numpy as np
import torch


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


def transfer_layer_scale(pt_ls, keras_ls):
    """Transfer LayerScale gamma parameter.

    Args:
        pt_ls: PyTorch LayerScale module.
        keras_ls: Keras LayerScale layer.
    """
    keras_ls.gamma.assign(to_keras(pt_ls.lambda1))


def transfer_patch_embeddings(pt_embed, keras_embed):
    """Transfer patch embedding weights including projection, CLS token,
    position embeddings, and register tokens.

    Args:
        pt_embed: PyTorch embedding module.
        keras_embed: Keras embedding layer.
    """
    transfer_conv2d(
        pt_embed.patch_embeddings.projection, keras_embed.projection
    )
    keras_embed.cls_token.assign(to_keras(pt_embed.cls_token))
    keras_embed.position_embeddings.assign(
        to_keras(pt_embed.position_embeddings)
    )
    if (
        pt_embed.register_tokens is not None
        and keras_embed.register_tokens is not None
    ):
        keras_embed.register_tokens.assign(
            to_keras(pt_embed.register_tokens)
        )


def transfer_attention(pt_attn, keras_attn):
    """Transfer attention weights, fusing separate Q/K/V into a single QKV.

    Args:
        pt_attn: PyTorch attention module with separate Q/K/V.
        keras_attn: Keras attention layer with fused QKV.
    """
    q_w = to_keras(pt_attn.attention.query.weight).T
    k_w = to_keras(pt_attn.attention.key.weight).T
    v_w = to_keras(pt_attn.attention.value.weight).T
    q_b = to_keras(pt_attn.attention.query.bias)
    k_b = to_keras(pt_attn.attention.key.bias)
    v_b = to_keras(pt_attn.attention.value.bias)

    fused_w = np.concatenate([q_w, k_w, v_w], axis=1)
    fused_b = np.concatenate([q_b, k_b, v_b], axis=0)
    keras_attn.predict_query_key_value.set_weights([fused_w, fused_b])

    transfer_dense(pt_attn.output.dense, keras_attn.projection_layer)


def transfer_mlp(pt_mlp, keras_mlp):
    """Transfer MLP (fc1/fc2) weights.

    Args:
        pt_mlp: PyTorch MLP module.
        keras_mlp: Keras MLP layer.
    """
    transfer_dense(pt_mlp.fc1, keras_mlp.fully_connected_layer_1)
    transfer_dense(pt_mlp.fc2, keras_mlp.fully_connected_layer_2)


def transfer_swiglu(pt_swiglu, keras_swiglu):
    """Transfer SwiGLU FFN weights.

    Args:
        pt_swiglu: PyTorch SwiGLU module.
        keras_swiglu: Keras SwiGLU layer.
    """
    transfer_dense(
        pt_swiglu.weights_in, keras_swiglu.fused_gate_and_value_projection
    )
    transfer_dense(pt_swiglu.weights_out, keras_swiglu.output_projection)


def transfer_layer(pt_layer, keras_layer):
    """Transfer all weights for a single DinoV2 encoder layer.

    Args:
        pt_layer: PyTorch encoder layer.
        keras_layer: Keras encoder layer.
    """
    transfer_layernorm(pt_layer.norm1, keras_layer.norm1)
    transfer_attention(pt_layer.attention, keras_layer.attention)
    transfer_layer_scale(pt_layer.layer_scale1, keras_layer.layer_scale1)
    transfer_layernorm(pt_layer.norm2, keras_layer.norm2)
    if hasattr(pt_layer.mlp, "fc1"):
        transfer_mlp(pt_layer.mlp, keras_layer.mlp)
    else:
        transfer_swiglu(pt_layer.mlp, keras_layer.mlp)
    transfer_layer_scale(pt_layer.layer_scale2, keras_layer.layer_scale2)


def transfer_encoder(pt_encoder, keras_encoder):
    """Transfer all layers of a DinoV2 encoder.

    Args:
        pt_encoder: PyTorch encoder with .layer attribute.
        keras_encoder: Keras encoder with .encoder_layers attribute.
    """
    for pt_l, k_l in zip(pt_encoder.layer, keras_encoder.encoder_layers):
        transfer_layer(pt_l, k_l)


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
    import keras
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
