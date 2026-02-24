import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import importlib
import sys

# Keras imports
import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import ops

# Ensure rf-detr PyTorch source is importable
rfdetr_parent = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../..", "rf-detr_original_pytorch_implementation"
))
if rfdetr_parent not in sys.path:
    sys.path.insert(0, rfdetr_parent)

# Ensure project root is in path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Keras modules under test
from examples.dino_v2_object_detection.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2PatchEmbeddings,
    WindowedDinov2Layer,
    WindowedDinov2Encoder,
    WindowedDinov2Model,
    dinov2_windowed_small,
    dinov2_windowed_base,
    dinov2_windowed_large,
    dinov2_windowed_giant,
)
from examples.dino_v2_object_detection.models.backbone.dinov2 import DinoV2

# PyTorch modules

from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersConfig,
    Dinov2WithRegistersPatchEmbeddings as PtPatchEmbed,
    WindowedDinov2WithRegistersEmbeddings as PtEmbeddings,
    Dinov2WithRegistersSelfAttention as PtSelfAttention,
    Dinov2WithRegistersSelfOutput as PtSelfOutput,
    Dinov2WithRegistersAttention as PtAttention,
    Dinov2WithRegistersLayerScale as PtLayerScale,
    Dinov2WithRegistersMLP as PtMLP,
    Dinov2WithRegistersSwiGLUFFN as PtSwiGLUFFN,
    WindowedDinov2WithRegistersLayer as PtLayer,
    WindowedDinov2WithRegistersEncoder as PtEncoder,
)

# ─── Weight porting & test helpers ───────────────────────────
from examples.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (
    ATOL,
    RTOL,
    assert_close,
    to_keras,
    chw_to_hwc,
    hwc_to_chw,
    build_keras_embed,
    transfer_conv2d,
    transfer_dense,
    transfer_layernorm,
    transfer_layer_scale,
    transfer_patch_embeddings,
    transfer_attention,
    transfer_mlp,
    transfer_swiglu,
    transfer_layer,
    transfer_encoder,
)

# ─── Helper: make a basic config ────────────────────────────
def make_pt_config(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    mlp_ratio=4,
    image_size=56,
    patch_size=14,
    num_register_tokens=0,
    num_windows=1,
    window_block_indexes=None,
    use_swiglu_ffn=False,
    layerscale_value=1.0,
    drop_path_rate=0.0,
):
    if window_block_indexes is None:
        window_block_indexes = list(range(num_hidden_layers))
    cfg = WindowedDinov2WithRegistersConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        mlp_ratio=mlp_ratio,
        image_size=image_size,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
        num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        use_swiglu_ffn=use_swiglu_ffn,
        layerscale_value=layerscale_value,
        drop_path_rate=drop_path_rate,
        hidden_act="gelu",
        out_features=[f"stage{num_hidden_layers}"],
        out_indices=[num_hidden_layers],
    )
    # HuggingFace PretrainedConfig normally sets this; we set it manually
    cfg._attn_implementation = "eager"
    return cfg



# ==============================================================
# TESTS
# ==============================================================

# --- Patch Embeddings ---

def test_patch_embeddings_output_shape():
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64, num_windows=1,
    )
    x = np.random.randn(2, 56, 56, 3).astype(np.float32)
    out = k(x, training=False)
    # (B, 1 + N_patches, C) = (2, 1 + 16, 64) = (2, 17, 64)
    assert ops.shape(out) == (2, 17, 64)


def test_patch_embeddings_parity():
    cfg = make_pt_config(hidden_size=64, image_size=56, patch_size=14)
    pt = PtEmbeddings(cfg).eval()

    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64, num_windows=1,
    )
    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    build_keras_embed(k, 1, 56, 56)
    transfer_patch_embeddings(pt, k)

    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    k_out = k(x_np, training=False)
    assert_close(pt_out, k_out)


def test_interpolate_pos_encoding_same_size():
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64, num_windows=1,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    k(x, training=False)

    dummy_emb = np.random.randn(1, 17, 64).astype(np.float32)
    result = k.interpolate_pos_encoding(dummy_emb, 56, 56)
    assert ops.shape(result) == (1, 17, 64)


def test_interpolate_pos_encoding_different_size():
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64, num_windows=1,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    k(x, training=False)

    # Target: 112x112 → 8x8=64 patches + CLS
    dummy_emb = np.random.randn(1, 65, 64).astype(np.float32)
    result = k.interpolate_pos_encoding(dummy_emb, 112, 112)
    assert ops.shape(result) == (1, 65, 64)


def test_register_tokens_insertion():
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64,
        num_register_tokens=4, num_windows=1,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = k(x, training=False)
    # CLS + 4 registers + 16 patches = 21
    assert ops.shape(out)[1] == 21


def test_no_register_tokens():
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64,
        num_register_tokens=0, num_windows=1,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = k(x, training=False)
    # CLS + 16 patches = 17
    assert ops.shape(out)[1] == 17


def test_register_tokens_parity():
    cfg = make_pt_config(
        hidden_size=64, image_size=56, patch_size=14, num_register_tokens=4,
    )
    pt = PtEmbeddings(cfg).eval()

    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64,
        num_register_tokens=4, num_windows=1,
    )
    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    build_keras_embed(k, 1, 56, 56)
    transfer_patch_embeddings(pt, k)

    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    k_out = k(x_np, training=False)
    assert_close(pt_out, k_out)


# --- Windowed Embeddings ---

def test_windowed_embeddings_shape():
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64,
        num_windows=2, num_register_tokens=0,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = k(x, training=False)
    # 2x2=4 windows, each with CLS + (2*2)=4 patches = 5
    # batch becomes 1*4 = 4
    assert ops.shape(out) == (4, 5, 64)


def test_windowed_embeddings_parity():
    cfg = make_pt_config(
        hidden_size=64, image_size=56, patch_size=14,
        num_windows=2, num_register_tokens=0,
    )
    pt = PtEmbeddings(cfg).eval()

    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64,
        num_windows=2, num_register_tokens=0,
    )
    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    build_keras_embed(k, 1, 56, 56)
    transfer_patch_embeddings(pt, k)

    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    k_out = k(x_np, training=False)
    assert_close(pt_out, k_out)


def test_windowed_with_registers_parity():
    cfg = make_pt_config(
        hidden_size=64, image_size=56, patch_size=14,
        num_windows=2, num_register_tokens=4,
    )
    pt = PtEmbeddings(cfg).eval()

    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64,
        num_windows=2, num_register_tokens=4,
    )
    x_np = np.random.randn(1, 56, 56, 3).astype(np.float32)
    build_keras_embed(k, 1, 56, 56)
    transfer_patch_embeddings(pt, k)

    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    with torch.no_grad():
        pt_out = pt(x_pt)
    k_out = k(x_np, training=False)
    assert_close(pt_out, k_out)


# --- LayerScale ---

def test_layer_scale_parity():
    from paz.models.foundation.dinov2.layers import LayerScale as KLayerScale
    from examples.dino_v2_object_detection.models.backbone.dinov2_with_windowed_attn import (
        WindowedDinov2Layer,
    )
    cfg = make_pt_config(hidden_size=64, layerscale_value=0.5)
    pt_ls = PtLayerScale(cfg).eval()

    k_ls = KLayerScale(64, init_values=0.5, name="test_ls")
    dummy = np.zeros((1, 4, 64), dtype=np.float32)
    k_ls(dummy)

    k_ls.gamma.assign(to_keras(pt_ls.lambda1))

    x_np = np.random.randn(1, 4, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_ls(x_pt)
    k_out = k_ls(x_np)
    assert_close(pt_out, k_out)


# --- MLP ---

def test_mlp_parity():
    cfg = make_pt_config(hidden_size=64, mlp_ratio=4)
    pt_mlp = PtMLP(cfg).eval()

    from paz.models.foundation.dinov2.layers import MLP as KMLP
    k_mlp = KMLP(
        input_features=64,
        hidden_features=256,
        activation_layer=keras.layers.Activation("gelu"),
        use_bias=True,
    )
    dummy = np.zeros((1, 4, 64), dtype=np.float32)
    k_mlp(dummy)

    transfer_mlp(pt_mlp, k_mlp)

    x_np = np.random.randn(1, 4, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_mlp(x_pt)
    k_out = k_mlp(x_np, training=False)
    assert_close(pt_out, k_out)


# --- SwiGLU FFN ---

def test_swiglu_ffn_parity():
    cfg = make_pt_config(hidden_size=64, mlp_ratio=4, use_swiglu_ffn=True)
    pt_swiglu = PtSwiGLUFFN(cfg).eval()

    from paz.models.foundation.dinov2.layers import SwiGLUFFNFused as KSwiGLU
    hidden_features = int(64 * 4)
    k_swiglu = KSwiGLU(
        input_features=64,
        hidden_features=hidden_features,
        use_bias=True,
        name="test_swiglu",
    )
    dummy = np.zeros((1, 4, 64), dtype=np.float32)
    k_swiglu(dummy)

    transfer_swiglu(pt_swiglu, k_swiglu)

    x_np = np.random.randn(1, 4, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_swiglu(x_pt)
    k_out = k_swiglu(x_np, training=False)
    assert_close(pt_out, k_out)


# --- Attention ---

def test_attention_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4)
    pt_att = PtAttention(cfg).eval()

    from paz.models.foundation.dinov2.layers import Attention as KAttention
    k_att = KAttention(
        dimension=64,
        number_of_heads=4,
        use_query_key_value_bias=True,
        use_projection_bias=True,
        name="test_attn",
    )
    dummy = np.zeros((1, 8, 64), dtype=np.float32)
    k_att(dummy)

    transfer_attention(pt_att, k_att)

    x_np = np.random.randn(1, 8, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_att(x_pt)[0]
    k_out = k_att(x_np, training=False)
    assert_close(pt_out, k_out)


# --- Drop Path (eval mode → identity) ---

def test_drop_path_eval_identity():
    from paz.models.foundation.dinov2.layers import DropPath as KDropPath
    k_dp = KDropPath(0.1, name="test_dp")
    x_np = np.random.randn(2, 4, 64).astype(np.float32)
    out = k_dp(x_np, training=False)
    np.testing.assert_allclose(np.array(out), x_np, atol=1e-7)


# --- Single Layer ---

def test_layer_no_windowing_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4, num_windows=1)
    pt_l = PtLayer(cfg).eval()

    k_l = WindowedDinov2Layer(
        hidden_size=64, num_attention_heads=4,
        mlp_ratio=4.0, num_windows=1, init_values=1.0,
    )
    dummy = np.zeros((1, 17, 64), dtype=np.float32)
    k_l(dummy, training=False)

    transfer_layer(pt_l, k_l)

    x_np = np.random.randn(1, 17, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_l(x_pt, output_attentions=False, run_full_attention=False)[0]
    k_out = k_l(x_np, training=False, run_full_attention=False)
    assert_close(pt_out, k_out)


def test_layer_with_full_attention_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4, num_windows=2)
    pt_l = PtLayer(cfg).eval()

    k_l = WindowedDinov2Layer(
        hidden_size=64, num_attention_heads=4,
        mlp_ratio=4.0, num_windows=2, init_values=1.0,
    )
    # With num_windows=2, batch is B*4, each window has 5 tokens
    dummy = np.zeros((4, 5, 64), dtype=np.float32)
    k_l(dummy, training=False, run_full_attention=True)

    transfer_layer(pt_l, k_l)

    x_np = np.random.randn(4, 5, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_l(x_pt, output_attentions=False, run_full_attention=True)[0]
    k_out = k_l(x_np, training=False, run_full_attention=True)
    assert_close(pt_out, k_out)


def test_layer_windowed_attention_parity():
    cfg = make_pt_config(hidden_size=64, num_attention_heads=4, num_windows=2)
    pt_l = PtLayer(cfg).eval()

    k_l = WindowedDinov2Layer(
        hidden_size=64, num_attention_heads=4,
        mlp_ratio=4.0, num_windows=2, init_values=1.0,
    )
    dummy = np.zeros((4, 5, 64), dtype=np.float32)
    k_l(dummy, training=False, run_full_attention=False)

    transfer_layer(pt_l, k_l)

    x_np = np.random.randn(4, 5, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_l(x_pt, output_attentions=False, run_full_attention=False)[0]
    k_out = k_l(x_np, training=False, run_full_attention=False)
    assert_close(pt_out, k_out)


# --- Encoder ---

def test_encoder_parity():
    cfg = make_pt_config(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_windows=1, window_block_indexes=[0, 1, 2],
    )
    pt_enc = PtEncoder(cfg).eval()

    k_enc = WindowedDinov2Encoder(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        mlp_ratio=4.0, num_windows=1,
        window_block_indexes=[0, 1, 2], init_values=1.0,
    )
    dummy = np.zeros((1, 17, 64), dtype=np.float32)
    k_enc(dummy, training=False)

    transfer_encoder(pt_enc, k_enc)

    x_np = np.random.randn(1, 17, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_enc(x_pt, output_hidden_states=True, return_dict=True)
    k_out = k_enc(x_np, training=False)

    # Compare final output
    assert_close(pt_out.last_hidden_state, k_out[-1])
    # Compare each intermediate
    for pt_h, k_h in zip(pt_out.hidden_states[1:], k_out):
        assert_close(pt_h, k_h)


def test_encoder_mixed_windowing_parity():
    cfg = make_pt_config(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_windows=2, window_block_indexes=[0, 1],
    )
    pt_enc = PtEncoder(cfg).eval()

    k_enc = WindowedDinov2Encoder(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
        mlp_ratio=4.0, num_windows=2,
        window_block_indexes=[0, 1], init_values=1.0,
    )
    # num_windows=2, each window has 5 tokens, batch*4
    dummy = np.zeros((4, 5, 64), dtype=np.float32)
    k_enc(dummy, training=False)

    transfer_encoder(pt_enc, k_enc)

    x_np = np.random.randn(4, 5, 64).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    with torch.no_grad():
        pt_out = pt_enc(x_pt, output_hidden_states=True, return_dict=True)
    k_out = k_enc(x_np, training=False)

    assert_close(pt_out.last_hidden_state, k_out[-1])


# --- Full Model ---

def test_model_output_shape():
    k_model = WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_windows=1, num_register_tokens=0,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    seq_out, all_hidden = k_model(x, training=False)
    # seq_out: (1, 17, 64), all_hidden: list of 2
    assert ops.shape(seq_out) == (1, 17, 64)
    assert len(all_hidden) == 2


def test_model_with_registers_shape():
    k_model = WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_windows=1, num_register_tokens=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    seq_out, _ = k_model(x, training=False)
    # CLS + 4 registers + 16 patches = 21
    assert ops.shape(seq_out) == (1, 21, 64)


def test_model_windowed_shape():
    k_model = WindowedDinov2Model(
        image_size=56, patch_size=14, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_windows=2, num_register_tokens=0,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    seq_out, _ = k_model(x, training=False)
    # Windowed: batch*4=4, CLS + 4 patches = 5
    assert ops.shape(seq_out) == (4, 5, 64)


# --- Builder functions ---

def test_builder_dinov2_windowed_small():
    model = dinov2_windowed_small(img_size=56, num_windows=1)
    assert model.hidden_size == 384


def test_builder_dinov2_windowed_base():
    model = dinov2_windowed_base(img_size=56, num_windows=1)
    assert model.hidden_size == 768


def test_builder_dinov2_windowed_large():
    model = dinov2_windowed_large(img_size=56, num_windows=1)
    assert model.hidden_size == 1024


def test_builder_dinov2_windowed_giant():
    model = dinov2_windowed_giant(img_size=56, num_windows=1)
    assert model.hidden_size == 1536


# --- DinoV2 Wrapper ---

def test_dinov2_wrapper_output_shapes():
    wrapper = DinoV2(
        shape=(56, 56),
        out_feature_indexes=[0, 1],
        size="small",
        use_registers=False,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    outputs = wrapper(x, training=False)
    assert len(outputs) == 2
    # Each: (1, 4, 4, 384)
    for out in outputs:
        assert ops.shape(out)[1] == 4
        assert ops.shape(out)[2] == 4
        assert ops.shape(out)[3] == 384


def test_dinov2_wrapper_with_registers_shapes():
    wrapper = DinoV2(
        shape=(56, 56),
        out_feature_indexes=[0, 1],
        size="small",
        use_registers=True,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    outputs = wrapper(x, training=False)
    assert len(outputs) == 2
    for out in outputs:
        assert ops.shape(out) == (1, 4, 4, 384)


def test_dinov2_wrapper_windowed_shapes():
    wrapper = DinoV2(
        shape=(56, 56),
        out_feature_indexes=[0, 1],
        size="small",
        use_registers=False,
        patch_size=14,
        num_windows=2,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    outputs = wrapper(x, training=False)
    assert len(outputs) == 2
    for out in outputs:
        assert ops.shape(out) == (1, 4, 4, 384)


# --- Different batch sizes ---

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(batch_size):
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64, num_windows=1,
    )
    x = np.random.randn(batch_size, 56, 56, 3).astype(np.float32)
    out = k(x, training=False)
    assert ops.shape(out)[0] == batch_size


@pytest.mark.parametrize("batch_size", [1, 2])
def test_different_batch_sizes_windowed(batch_size):
    k = WindowedDinov2PatchEmbeddings(
        image_size=56, patch_size=14, hidden_size=64, num_windows=2,
    )
    x = np.random.randn(batch_size, 56, 56, 3).astype(np.float32)
    out = k(x, training=False)
    assert ops.shape(out)[0] == batch_size * 4


# --- Config loading ---

def test_dinov2_small_config():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1],
        size="small", use_registers=True,
        patch_size=14, num_windows=1, positional_encoding_size=4,
    )
    assert wrapper.encoder.hidden_size == 384


def test_dinov2_base_config():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1],
        size="base", use_registers=True,
        patch_size=14, num_windows=1, positional_encoding_size=4,
    )
    assert wrapper.encoder.hidden_size == 768


def test_dinov2_large_config():
    wrapper = DinoV2(
        shape=(56, 56), out_feature_indexes=[0, 1],
        size="large", use_registers=True,
        patch_size=14, num_windows=1, positional_encoding_size=4,
    )
    assert wrapper.encoder.hidden_size == 1024




if __name__ == "__main__":
    pytest.main([__file__])