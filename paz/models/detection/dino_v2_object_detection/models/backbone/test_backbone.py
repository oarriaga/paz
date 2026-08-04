import functools
import os
import sys

import numpy as np
import pytest

_PT_DIR = "examples/rf-detr_original_pytorch_implementation"
_PT_ROOT = os.path.join(os.path.dirname(__file__), *([".."] * 6))
if not os.path.isdir(os.path.join(_PT_ROOT, _PT_DIR)):
    pytest.skip("RF-DETR reference unavailable", allow_module_level=True)
pytest.importorskip("torch")

import torch

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import ops

rfdetr_parent = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "..",
    "examples", "rf-detr_original_pytorch_implementation"
))
if rfdetr_parent not in sys.path:
    sys.path.insert(0, rfdetr_parent)

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from paz.models.detection.dino_v2_object_detection.models.backbone import (
    position_embedding_sine as k_position_embedding_sine,
    build_position_encoding as k_build_position_encoding,
    Backbone as KBackbone,
    get_dinov2_lr_decay_rate as k_get_dinov2_lr_decay_rate,
    get_dinov2_weight_decay_rate as k_get_dinov2_weight_decay_rate,
    build_backbone as k_build_backbone,
)

from rfdetr.models.position_encoding import (
    PositionEmbeddingSine as PtPositionEmbeddingSine,
)
from rfdetr.models.backbone.backbone import (
    get_dinov2_lr_decay_rate as pt_get_dinov2_lr_decay_rate,
    get_dinov2_weight_decay_rate as pt_get_dinov2_weight_decay_rate,
)

from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (  # fmt: skip
    assert_close,
    make_mask,
    make_pt_nested_tensor,
)


def test_pos_sine_output_shape_aligned():
    mask = np.zeros((2, 8, 10), dtype=bool)
    out = k_position_embedding_sine(
        mask, num_pos_feats=32, normalize=True, align_dim_orders=True
    )
    assert ops.shape(out) == (8, 10, 2, 64)


def test_pos_sine_output_shape_default():
    mask = np.zeros((2, 8, 10), dtype=bool)
    out = k_position_embedding_sine(
        mask, num_pos_feats=32, normalize=True, align_dim_orders=False
    )
    assert ops.shape(out) == (2, 8, 10, 64)


def test_pos_sine_parity_aligned():
    pt = PtPositionEmbeddingSine(num_pos_feats=32, normalize=True).eval()
    mask_np = make_mask(1, 6, 8, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 6, 8, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=True)
    k_out = k_position_embedding_sine(
        mask_np, num_pos_feats=32, normalize=True, align_dim_orders=True
    )
    assert_close(pt_out, k_out)


def test_pos_sine_parity_channel_first():
    pt = PtPositionEmbeddingSine(num_pos_feats=64, normalize=True).eval()
    mask_np = make_mask(2, 10, 12, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(2, 10, 12, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        pt_out = pt_out.permute(0, 2, 3, 1)
    k_out = k_position_embedding_sine(
        mask_np, num_pos_feats=64, normalize=True, align_dim_orders=False
    )
    assert_close(pt_out, k_out)


def test_pos_sine_parity_with_masking():
    pt = PtPositionEmbeddingSine(num_pos_feats=32, normalize=True).eval()
    mask_np = make_mask(1, 6, 8, all_false=False)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 6, 8, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        pt_out = pt_out.permute(0, 2, 3, 1)
    k_out = k_position_embedding_sine(
        mask_np, num_pos_feats=32, normalize=True, align_dim_orders=False
    )
    assert_close(pt_out, k_out)


def test_pos_sine_no_normalize():
    pt = PtPositionEmbeddingSine(num_pos_feats=16, normalize=False).eval()
    mask_np = make_mask(1, 4, 4, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 4, 4, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        pt_out = pt_out.permute(0, 2, 3, 1)
    k_out = k_position_embedding_sine(
        mask_np, num_pos_feats=16, normalize=False, align_dim_orders=False
    )
    assert_close(pt_out, k_out)


def test_pos_sine_export_parity():
    pt = PtPositionEmbeddingSine(num_pos_feats=32, normalize=True).eval()
    mask_np = make_mask(2, 5, 7, all_false=True)
    mask_pt = torch.from_numpy(mask_np)
    with torch.no_grad():
        pt_out = pt.forward_export(mask_pt, align_dim_orders=False)
        pt_out = pt_out.permute(0, 2, 3, 1)
    k_out = k_position_embedding_sine(
        mask_np, num_pos_feats=32, normalize=True, align_dim_orders=False
    )
    assert_close(pt_out, k_out)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_pos_sine_different_batch_sizes(batch_size):
    mask = np.zeros((batch_size, 6, 6), dtype=bool)
    out = k_position_embedding_sine(
        mask, num_pos_feats=32, normalize=True, align_dim_orders=False
    )
    assert ops.shape(out)[0] == batch_size


@pytest.mark.parametrize("h,w", [(4, 4), (8, 6), (16, 16), (3, 7)])
def test_pos_sine_different_spatial_sizes(h, w):
    mask = np.zeros((1, h, w), dtype=bool)
    out = k_position_embedding_sine(
        mask, num_pos_feats=32, normalize=True, align_dim_orders=False
    )
    assert ops.shape(out) == (1, h, w, 64)


def test_pos_sine_scale_no_normalize_raises():
    mask = np.zeros((1, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="normalize should be True"):
        k_position_embedding_sine(
            mask, num_pos_feats=32, normalize=False, scale=1.0
        )


def test_pos_sine_custom_temperature_parity():
    pt = PtPositionEmbeddingSine(
        num_pos_feats=16, temperature=5000, normalize=True
    ).eval()
    mask_np = make_mask(1, 4, 6, all_false=True)
    nt = make_pt_nested_tensor(
        np.random.randn(1, 4, 6, 3).astype(np.float32), mask_np
    )
    with torch.no_grad():
        pt_out = pt(nt, align_dim_orders=False)
        pt_out = pt_out.permute(0, 2, 3, 1)
    k_out = k_position_embedding_sine(
        mask_np, num_pos_feats=16, temperature=5000, normalize=True,
        align_dim_orders=False,
    )
    assert_close(pt_out, k_out)


def test_build_pos_encoding_sine():
    pe = k_build_position_encoding(256, "sine")
    assert isinstance(pe, functools.partial)
    assert pe.func is k_position_embedding_sine
    assert pe.keywords["num_pos_feats"] == 128


def test_build_pos_encoding_v2():
    pe = k_build_position_encoding(512, "v2")
    assert isinstance(pe, functools.partial)
    assert pe.keywords["num_pos_feats"] == 256


def test_build_pos_encoding_unsupported():
    with pytest.raises(ValueError):
        k_build_position_encoding(256, "unknown")


def test_build_pos_encoding_normalize():
    pe = k_build_position_encoding(128, "sine")
    assert pe.keywords["normalize"] is True


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
    kwargs = dict(lr_decay_rate=0.85, num_layers=12)
    pt_val = pt_get_dinov2_lr_decay_rate(name, **kwargs)
    k_val = k_get_dinov2_lr_decay_rate(name, **kwargs)
    assert k_val == pytest.approx(pt_val)


def test_wd_gamma():
    assert k_get_dinov2_weight_decay_rate("layer.gamma") == 0.0
    assert pt_get_dinov2_weight_decay_rate("layer.gamma") == 0.0


def test_wd_pos_embed():
    assert k_get_dinov2_weight_decay_rate("pos_embed") == 0.0


def test_wd_rel_pos():
    assert k_get_dinov2_weight_decay_rate("rel_pos_h") == 0.0


def test_wd_bias():
    name = "encoder.layer.0.attention.bias"
    assert k_get_dinov2_weight_decay_rate(name) == 0.0


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


def test_backbone_name_base():
    b = KBackbone(
        name="dinov2_base",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    assert b.get_layer("encoder").size == "base"


def test_backbone_name_registers():
    b = KBackbone(
        name="dinov2_registers_small",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )
    assert b.get_layer("encoder").use_registers is True
    assert b.get_layer("encoder").size == "small"


def test_backbone_name_windowed():
    b = KBackbone(
        name="dinov2_windowed_base",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=2,
        positional_encoding_size=4,
    )
    assert b.get_layer("encoder").size == "base"


def test_backbone_name_registers_windowed():
    b = KBackbone(
        name="dinov2_registers_windowed_large",
        out_feature_indexes=[0, 1],
        projector_scale=["P4"],
        target_shape=(56, 56),
        patch_size=14,
        num_windows=2,
        positional_encoding_size=4,
    )
    assert b.get_layer("encoder").use_registers is True
    assert b.get_layer("encoder").size == "large"


def test_backbone_name_invalid():
    with pytest.raises(AssertionError):
        KBackbone(
            name="resnet50",
            out_feature_indexes=[0, 1],
            projector_scale=["P4"],
        )


def test_backbone_projector_scale_order():
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


def test_backbone_forward_output_count():
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
    out = b([x, mask], training=False)
    assert len(out) == 2


def test_backbone_forward_returns_tuples():
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
    mask = np.zeros((1, 56, 56), dtype=bool)
    out = b([x, mask], training=False)
    assert len(out) == 1
    feat, m = out[0]
    assert len(ops.shape(feat)) == 4
    assert len(ops.shape(m)) == 3


def test_backbone_forward_channels():
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
    mask = np.zeros((1, 56, 56), dtype=bool)
    out = b([x, mask], training=False)
    feat, _ = out[0]
    assert ops.shape(feat)[3] == 128


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
    mask = np.zeros((batch_size, 56, 56), dtype=bool)
    out = b([x, mask], training=False)
    feat, m = out[0]
    assert ops.shape(feat)[0] == batch_size
    assert ops.shape(m)[0] == batch_size


def test_backbone_p3_upsamples():
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
    out = b([x, mask], training=False)
    p3_h = ops.shape(out[0][0])[2]
    p4_h = ops.shape(out[1][0])[2]
    assert p3_h > p4_h


def test_backbone_output_channels_match_out_channels():
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
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    out = b([x, mask], training=False)
    assert len(out) == 2
    for feat, _ in out:
        assert ops.shape(feat)[3] == 128


def build_small_joiner(projector_scale, hidden_dim=64):
    return k_build_backbone(
        encoder="dinov2_small",
        out_feature_indexes=[0, 1],
        projector_scale=projector_scale,
        target_shape=(56, 56),
        out_channels=64,
        hidden_dim=hidden_dim,
        position_embedding="sine",
        patch_size=14,
        num_windows=1,
        positional_encoding_size=4,
    )


def test_build_backbone_returns_joiner():
    model = build_small_joiner(["P4"])
    assert isinstance(model, keras.Model)
    assert model.name == "joiner"


def test_build_backbone_output_structure():
    model = build_small_joiner(["P4"])
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, positions = model([x, mask], training=False)
    assert len(features) == 1
    assert len(positions) == 1
    feat, m = features[0]
    assert len(ops.shape(feat)) == 4
    assert len(ops.shape(m)) == 3


def test_build_backbone_pos_shape():
    model = build_small_joiner(["P4"], hidden_dim=64)
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, positions = model([x, mask], training=False)
    feat, _ = features[0]
    pos = positions[0]
    assert ops.shape(pos)[0] == ops.shape(feat)[0]
    assert ops.shape(pos)[1] == ops.shape(feat)[1]
    assert ops.shape(pos)[2] == ops.shape(feat)[2]
    assert ops.shape(pos)[3] == 64


def test_build_backbone_end_to_end():
    model = build_small_joiner(["P4"])
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, positions = model([x, mask], training=False)
    assert len(features) == 1
    assert len(positions) == 1


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
    assert model.name == "joiner"


def test_build_backbone_multi_scale():
    model = build_small_joiner(["P3", "P4"])
    x = np.random.randn(1, 56, 56, 3).astype(np.float32)
    mask = np.zeros((1, 56, 56), dtype=bool)
    features, positions = model([x, mask], training=False)
    assert len(features) == 2
    assert len(positions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
