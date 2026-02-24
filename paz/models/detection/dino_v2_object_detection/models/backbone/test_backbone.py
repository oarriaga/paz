import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import sys
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import ops

# Ensure rf-detr PyTorch source is importable
rfdetr_parent = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..","..", "..", "..", "..", "..", "examples", "rf-detr_original_pytorch_implementation"
))
if rfdetr_parent not in sys.path:
    sys.path.insert(0, rfdetr_parent)

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ─── Keras imports ──────────────────────────────────────────────────
from paz.models.detection.dino_v2_object_detection.models.backbone.position_encoding import (
    PositionEmbeddingSine as KPositionEmbeddingSine,
    build_position_encoding as k_build_position_encoding,
)
from paz.models.detection.dino_v2_object_detection.models.backbone.backbone import (
    Backbone as KBackbone,
    get_dinov2_lr_decay_rate as k_get_dinov2_lr_decay_rate,
    get_dinov2_weight_decay_rate as k_get_dinov2_weight_decay_rate,
)
from paz.models.detection.dino_v2_object_detection.models.backbone import (
    Joiner as KJoiner,
    build_backbone as k_build_backbone,
)

# ─── PyTorch imports ────────────────────────────────────────────────
from rfdetr.models.position_encoding import (
    PositionEmbeddingSine as PtPositionEmbeddingSine,
    build_position_encoding as pt_build_position_encoding,
)
from rfdetr.models.backbone.backbone import (
    get_dinov2_lr_decay_rate as pt_get_dinov2_lr_decay_rate,
    get_dinov2_weight_decay_rate as pt_get_dinov2_weight_decay_rate,
)

# ─── Tolerances & helpers from shared utils ─────────────────────────
from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (
    ATOL,
    RTOL,
    assert_close,
    make_mask,
    make_pt_nested_tensor,
)



# ===========================================================================
# PositionEmbeddingSine Tests
# ===========================================================================

def test_pos_sine_output_shape_aligned():
    """Output shape (H, W, B, C) when align_dim_orders=True."""
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask = np.zeros((2, 8, 10), dtype=bool)
    out = k(mask, align_dim_orders=True)
    assert ops.shape(out) == (8, 10, 2, 64)


def test_pos_sine_output_shape_default():
    """Output shape (B, H, W, C) when align_dim_orders=False (default)."""
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask = np.zeros((2, 8, 10), dtype=bool)
    out = k(mask, align_dim_orders=False)
    assert ops.shape(out) == (2, 8, 10, 64)


def test_pos_sine_parity_aligned():
    """Numerical parity with PyTorch, align_dim_orders=True."""
    pt = PtPositionEmbeddingSine(num_pos_feats=32, normalize=True).eval()
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask_np = make_mask(1, 6, 8, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 6, 8, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=True)
    k_out = k(mask_np, align_dim_orders=True)
    assert_close(pt_out, k_out)


def test_pos_sine_parity_channel_first():
    """Numerical parity with PyTorch, align_dim_orders=False."""
    pt = PtPositionEmbeddingSine(num_pos_feats=64, normalize=True).eval()
    k = KPositionEmbeddingSine(num_pos_feats=64, normalize=True)
    mask_np = make_mask(2, 10, 12, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(2, 10, 12, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        # PT: (B, C, H, W) -> (B, H, W, C) for comparison
        pt_out = pt_out.permute(0, 2, 3, 1)
        
    k_out = k(mask_np, align_dim_orders=False)
    assert_close(pt_out, k_out)


def test_pos_sine_parity_with_masking():
    """Parity when mask has True values (padding regions)."""
    pt = PtPositionEmbeddingSine(num_pos_feats=32, normalize=True).eval()
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask_np = make_mask(1, 6, 8, all_false=False)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 6, 8, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        # PT: (B, C, H, W) -> (B, H, W, C)
        pt_out = pt_out.permute(0, 2, 3, 1)

    k_out = k(mask_np, align_dim_orders=False)
    assert_close(pt_out, k_out)


def test_pos_sine_no_normalize():
    """Parity without normalization."""
    pt = PtPositionEmbeddingSine(num_pos_feats=16, normalize=False).eval()
    k = KPositionEmbeddingSine(num_pos_feats=16, normalize=False)
    mask_np = make_mask(1, 4, 4, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 4, 4, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        # PT: (B, C, H, W) -> (B, H, W, C)
        pt_out = pt_out.permute(0, 2, 3, 1)

    k_out = k(mask_np, align_dim_orders=False)
    assert_close(pt_out, k_out)


def test_pos_sine_export_parity():
    """Export mode parity (mask-only input)."""
    pt = PtPositionEmbeddingSine(num_pos_feats=32, normalize=True).eval()
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask_np = make_mask(2, 5, 7, all_false=True)
    mask_pt = torch.from_numpy(mask_np)
    with torch.no_grad():
        pt_out = pt.forward_export(mask_pt, align_dim_orders=False)
        # PT: (B, C, H, W) -> (B, H, W, C)
        pt_out = pt_out.permute(0, 2, 3, 1)

    k_out = k(mask_np, align_dim_orders=False)
    assert_close(pt_out, k_out)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_pos_sine_different_batch_sizes(batch_size):
    """Handles different batch sizes correctly."""
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask = np.zeros((batch_size, 6, 6), dtype=bool)
    out = k(mask, align_dim_orders=False)
    assert ops.shape(out)[0] == batch_size


@pytest.mark.parametrize("h,w", [(4, 4), (8, 6), (16, 16), (3, 7)])
def test_pos_sine_different_spatial_sizes(h, w):
    """Handles various H, W."""
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    mask = np.zeros((1, h, w), dtype=bool)
    out = k(mask, align_dim_orders=False)
    assert ops.shape(out) == (1, h, w, 64)


def test_pos_sine_scale_no_normalize_raises():
    """Passing scale without normalize should raise."""
    with pytest.raises(ValueError, match="normalize should be True"):
        KPositionEmbeddingSine(num_pos_feats=32, normalize=False, scale=1.0)


def test_pos_sine_custom_temperature_parity():
    """Parity with custom temperature value."""
    pt = PtPositionEmbeddingSine(num_pos_feats=16, temperature=5000, normalize=True).eval()
    k = KPositionEmbeddingSine(num_pos_feats=16, temperature=5000, normalize=True)
    mask_np = make_mask(1, 4, 6, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 4, 6, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        pt_out = pt_out.permute(0, 2, 3, 1)

    k_out = k(mask_np, align_dim_orders=False)
    assert_close(pt_out, k_out)


def test_pos_sine_tuple_input():
    """Supports (tensors, mask) as separate args."""
    k = KPositionEmbeddingSine(num_pos_feats=32, normalize=True)
    im = np.zeros((1, 6, 6, 3), dtype=np.float32)
    mask = np.zeros((1, 6, 6), dtype=bool)
    out = k(im, mask=mask, align_dim_orders=False)
    assert ops.shape(out) == (1, 6, 6, 64)


def test_pos_sine_get_config():
    """Config round-trip."""
    k = KPositionEmbeddingSine(
        num_pos_feats=48, temperature=8000, normalize=True, scale=3.0
    )
    cfg = k.get_config()
    assert cfg["num_pos_feats"] == 48
    assert cfg["temperature"] == 8000
    assert cfg["normalize"] is True
    assert cfg["scale"] == 3.0


# ===========================================================================
# build_position_encoding Tests
# ===========================================================================

def test_build_pos_encoding_sine():
    pe = k_build_position_encoding(256, "sine")
    assert isinstance(pe, KPositionEmbeddingSine)
    assert pe.num_pos_feats == 128


def test_build_pos_encoding_v2():
    pe = k_build_position_encoding(512, "v2")
    assert isinstance(pe, KPositionEmbeddingSine)
    assert pe.num_pos_feats == 256


def test_build_pos_encoding_unsupported():
    with pytest.raises(ValueError):
        k_build_position_encoding(256, "unknown")


def test_build_pos_encoding_normalize():
    pe = k_build_position_encoding(128, "sine")
    assert pe.normalize is True


# ===========================================================================
# get_dinov2_lr_decay_rate Tests
# ===========================================================================

def test_lr_decay_embeddings():
    name = "backbone.0.encoder.embeddings.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    assert k_val == pytest.approx(pt_val)


def test_lr_decay_layer_3():
    name = "backbone.0.encoder.layer.3.attention.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    assert k_val == pytest.approx(pt_val)


def test_lr_decay_layer_0():
    name = "backbone.0.encoder.layer.0.mlp.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.8, num_layers=6)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.8, num_layers=6)
    assert k_val == pytest.approx(pt_val)


def test_lr_decay_last_layer():
    name = "backbone.0.encoder.layer.11.norm.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    assert k_val == pytest.approx(pt_val)


def test_lr_decay_residual_excluded():
    name = "backbone.0.encoder.layer.5.residual.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    assert k_val == pytest.approx(pt_val)


def test_lr_decay_non_backbone():
    name = "decoder.layer.3.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.9, num_layers=12)
    assert k_val == pytest.approx(pt_val)


def test_lr_decay_rate_1():
    name = "backbone.0.encoder.layer.5.weight"
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12)
    assert k_val == 1.0


@pytest.mark.parametrize("layer_id", [0, 1, 5, 11])
def test_lr_decay_all_layers(layer_id):
    name = f"backbone.0.encoder.layer.{layer_id}.attention.weight"
    pt_val = pt_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.85, num_layers=12)
    k_val = k_get_dinov2_lr_decay_rate(name, lr_decay_rate=0.85, num_layers=12)
    assert k_val == pytest.approx(pt_val)


# ===========================================================================
# get_dinov2_weight_decay_rate Tests
# ===========================================================================

def test_wd_gamma():
    assert k_get_dinov2_weight_decay_rate("layer.gamma") == 0.0
    assert pt_get_dinov2_weight_decay_rate("layer.gamma") == 0.0


def test_wd_pos_embed():
    assert k_get_dinov2_weight_decay_rate("pos_embed") == 0.0


def test_wd_rel_pos():
    assert k_get_dinov2_weight_decay_rate("rel_pos_h") == 0.0


def test_wd_bias():
    assert k_get_dinov2_weight_decay_rate("encoder.layer.0.attention.bias") == 0.0


def test_wd_norm():
    assert k_get_dinov2_weight_decay_rate("encoder.norm.weight") == 0.0


def test_wd_embeddings():
    assert k_get_dinov2_weight_decay_rate("encoder.embeddings.weight") == 0.0


def test_wd_regular_weight():
    val = k_get_dinov2_weight_decay_rate("encoder.layer.0.attention.weight")
    assert val == 1.0


@pytest.mark.parametrize("name,expected", [
    ("gamma_scale", 0.0),
    ("pos_embed_proj", 0.0),
    ("rel_pos_bias", 0.0),
    ("layer.bias", 0.0),
    ("norm1.weight", 0.0),
    ("embeddings.patch", 0.0),
    ("conv.weight", 1.0),
    ("fc.weight", 1.0),
])
def test_wd_parametrized(name, expected):
    pt_val = pt_get_dinov2_weight_decay_rate(name)
    k_val = k_get_dinov2_weight_decay_rate(name)
    assert k_val == expected
    assert pt_val == expected


# ===========================================================================
# Backbone Name Parsing Tests
# ===========================================================================

def test_backbone_name_base():
    """Parses 'dinov2_base' correctly."""
    b = KBackbone(
        name="dinov2_base",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    assert b.encoder.size == "base"


def test_backbone_name_registers():
    """Parses 'dinov2_registers_small' correctly."""
    b = KBackbone(
        name="dinov2_registers_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    assert b.encoder.use_registers is True
    assert b.encoder.size == "small"


def test_backbone_name_windowed():
    """Parses 'dinov2_windowed_base' correctly."""
    b = KBackbone(
        name="dinov2_windowed_base",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=2,
        positional_encoding_size=4,
    )
    assert b.encoder.size == "base"


def test_backbone_name_registers_windowed():
    """Parses 'dinov2_registers_windowed_large'."""
    b = KBackbone(
        name="dinov2_registers_windowed_large",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=2,
        positional_encoding_size=4,
    )
    assert b.encoder.use_registers is True
    assert b.encoder.size == "large"


def test_backbone_name_invalid():
    """Bad name raises AssertionError."""
    with pytest.raises(AssertionError):
        KBackbone(
            name="resnet50",
            out_feature_indexes=[0, 1],
            projector_scale=["P4"],
        )


def test_backbone_projector_scale_order():
    """Unsorted projector_scale raises AssertionError."""
    with pytest.raises(AssertionError):
        KBackbone(
            name="dinov2_base",
            out_feature_indexes=[0, 1],
            projector_scale=["P5", "P3"],
            target_shape=(56, 56),
            patch_size=14,
            num_windows=1,
            positional_encoding_size=4,
        )


# ===========================================================================
# Backbone Forward Shape Tests
# ===========================================================================

def test_backbone_forward_output_count():
    """Number of outputs matches projector_scale."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P3", "P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    out = b(x, mask=mask, training=False)
    assert len(out) == 2


def test_backbone_forward_returns_tuples():
    """Each output is a (feat, mask) tuple."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = b(x, training=False)
    assert len(out) == 1
    feat, m = out[0]
    assert len(ops.shape(feat)) == 4
    assert len(ops.shape(m)) == 3


def test_backbone_forward_channels():
    """Output channels match out_channels."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=128,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = b(x, training=False)
    feat, _ = out[0]
    assert ops.shape(feat)[3] == 128  # channels-last (B, H, W, C)


def test_backbone_export_returns_lists():
    """call_export returns (feats_list, masks_list)."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    feats, masks = b.call_export(x, training=False)
    assert len(feats) == 1
    assert len(masks) == 1


def test_backbone_export_mask_all_false():
    """Export masks are all-False."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    _, masks = b.call_export(x, training=False)
    assert not np.any(np.array(masks[0]))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_backbone_batch_sizes(batch_size):
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(batch_size, 56, 56, 3).astype(np.float32)
    out = b(x, training=False)
    feat, m = out[0]
    assert ops.shape(feat)[0] == batch_size
    assert ops.shape(m)[0] == batch_size


def test_backbone_p3_upsamples():
    """P3 scale (2.0) should produce larger spatial dims than P4 (1.0)."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P3", "P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    out = b(x, training=False)
    p3_h = ops.shape(out[0][0])[2]
    p4_h = ops.shape(out[1][0])[2]
    assert p3_h > p4_h


# ===========================================================================
# Joiner Tests
# ===========================================================================

def test_joiner_output_structure():
    """Joiner returns (x, pos) where x is list of (feat, mask)."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    pe = k_build_position_encoding(64, "sine")
    joiner = KJoiner(b, pe)

    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, pos = joiner(x, mask=mask, training=False)
    assert len(features) == 1
    assert len(pos) == 1


def test_joiner_pos_shape():
    """Position encoding has correct shape (B, C, H, W)."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    pe = k_build_position_encoding(64, "sine")
    joiner = KJoiner(b, pe)

    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, pos = joiner(x, mask=mask, training=False)
    feat, feat_mask = features[0]
    p = pos[0]
    # pos should be (B, C, H, W) since align_dim_orders=False
    assert ops.shape(p)[0] == ops.shape(feat)[0]  # batch
    assert ops.shape(p)[1] == ops.shape(feat)[1]  # height
    assert ops.shape(p)[2] == ops.shape(feat)[2]  # width
    assert ops.shape(p)[3] == 64 # num_pos_feats default (32*2)


def test_joiner_export_returns_three():
    """call_export returns (feats, None, poss)."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    pe = k_build_position_encoding(64, "sine")
    joiner = KJoiner(b, pe)

    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    result = joiner.call_export(x, training=False)
    feats, none_val, poss = result
    assert none_val is None
    assert len(feats) == 1
    assert len(poss) == 1


def test_joiner_multiple_scales():
    """Joiner handles multiple projector scales."""
    b = KBackbone(
        name="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P3", "P4"],
        target_shape=(56, 56),
        out_channels=64,
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    pe = k_build_position_encoding(64, "sine")
    joiner = KJoiner(b, pe)

    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, pos = joiner(x, mask=mask, training=False)
    assert len(features) == 2
    assert len(pos) == 2


# ===========================================================================
# build_backbone Tests
# ===========================================================================

def test_build_backbone_returns_joiner():
    model = k_build_backbone(
        encoder="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        hidden_dim=64,
        position_embedding="sine",
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    assert isinstance(model, KJoiner)


def test_build_backbone_end_to_end():
    model = k_build_backbone(
        encoder="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        hidden_dim=64,
        position_embedding="sine",
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, pos = model(x, mask=mask, training=False)
    assert len(features) == 1
    assert len(pos) == 1


def test_build_backbone_with_registers():
    model = k_build_backbone(
        encoder="dinov2_registers_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        out_channels=64,
        hidden_dim=64,
        position_embedding="sine",
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    assert isinstance(model, KJoiner)


def test_build_backbone_multi_scale():
    model = k_build_backbone(
        encoder="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P3", "P4"],
        target_shape=(56, 56),
        out_channels=64,
        hidden_dim=64,
        position_embedding="sine",
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, pos = model(x, mask=mask, training=False)
    assert len(features) == 2
    assert len(pos) == 2


# ===========================================================================
# Backbone get_config round-trip
# ===========================================================================

def test_backbone_get_config():
    b = KBackbone(
        name="dinov2_registers_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P3", "P4"],
        target_shape=(56, 56),
        out_channels=128,
        patch_size=14,
        num_windows=2,
        positional_encoding_size=4,
    )
    cfg = b.get_config()
    assert cfg["out_channels"] == 128
    assert cfg["projector_scale"] == ["P3", "P4"]
    assert cfg["target_shape"] == (56, 56)
