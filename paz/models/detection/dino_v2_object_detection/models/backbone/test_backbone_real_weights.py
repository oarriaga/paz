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

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
except ImportError:
    # Fallback to local source if not installed
    rfdetr_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..", "..", "..",
            "examples", "rf-detr_original_pytorch_implementation",
        )
    )
    if rfdetr_path not in sys.path:
        sys.path.insert(0, rfdetr_path)
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

from paz.models.detection.dino_v2_object_detection.models.backbone import (
    Backbone,
    build_backbone,
)
from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (  # fmt: skip
    transfer_encoder,
    transfer_layernorm,
    port_weights_multiscale_projector,
    optional_embedding_table,
    hwc_to_chw,
    chw_to_hwc,
)


# NOTE: PyTorch uses 1-based indexing for layers [3, 6, 9, 12].
# Keras DinoV2 (0-based) must use [2, 5, 8, 11] to extract the same features.
MODEL_CONFIGS = {
    "Nano": {
        "cls": RFDETRNano,
        "encoder_name": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 384,
        "hidden_dim": 256,
        "positional_encoding_size": 24,
        "projector_scale": ["P4"],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "layer_norm": True,
    },
    "Small": {
        "cls": RFDETRSmall,
        "encoder_name": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 512,
        "hidden_dim": 256,
        "positional_encoding_size": 32,
        "projector_scale": ["P4"],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "layer_norm": True,
    },
    "Medium": {
        "cls": RFDETRMedium,
        "encoder_name": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 576,
        "hidden_dim": 256,
        "positional_encoding_size": 36,
        "projector_scale": ["P4"],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "layer_norm": True,
    },
    "Large": {
        "cls": RFDETRLarge,
        "encoder_name": "dinov2_windowed_small",
        "out_feature_indexes": [2, 5, 8, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 704,
        "hidden_dim": 256,
        "positional_encoding_size": 44,
        "projector_scale": ["P4"],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "layer_norm": True,
    },
}


def _extract_pt_parts(model_class):
    wrapper = model_class(pretrained=True)
    core_model = wrapper.model.model
    core_model.eval()

    joiner = core_model.backbone
    pt_backbone = joiner[0]
    pt_pos_embed = joiner[1]
    return pt_backbone, pt_pos_embed, core_model


def _build_keras_backbone(cfg):
    res = cfg["resolution"]
    return Backbone(
        name=cfg["encoder_name"],
        out_feature_indexes=cfg["out_feature_indexes"],
        projector_scale=cfg["projector_scale"],
        patch_size=cfg["patch_size"],
        num_windows=cfg["num_windows"],
        window_block_indexes=cfg["window_block_indexes"],
        out_channels=cfg["hidden_dim"],
        positional_encoding_size=cfg["positional_encoding_size"],
        layer_norm=cfg.get("layer_norm", False),
        load_dinov2_weights=False,  # We load manually
        target_shape=(res, res),
    )


def _build_keras_joiner(cfg):
    res = cfg["resolution"]
    return build_backbone(
        encoder=cfg["encoder_name"],
        out_feature_indexes=cfg["out_feature_indexes"],
        patch_size=cfg["patch_size"],
        num_windows=cfg["num_windows"],
        window_block_indexes=cfg["window_block_indexes"],
        positional_encoding_size=cfg["positional_encoding_size"],
        projector_scale=cfg["projector_scale"],
        out_channels=cfg["hidden_dim"],
        hidden_dim=cfg["hidden_dim"],
        position_embedding="sine",
        target_shape=(res, res),
        layer_norm=cfg.get("layer_norm", False),
        load_dinov2_weights=False,
    )


def _resize_and_assign_pos_embed(pt_embeddings, keras_pos):
    if hasattr(pt_embeddings.position_embeddings, "weight"):
        pt_pos = pt_embeddings.position_embeddings.weight.detach().cpu().numpy()
    else:
        pt_pos = pt_embeddings.position_embeddings.detach().cpu().numpy()

    if pt_pos.ndim == 2:
        pt_pos = np.expand_dims(pt_pos, axis=0)

    keras_shape = keras_pos.shape

    if pt_pos.shape[1] == keras_shape[0]:
        keras_pos.assign(np.reshape(pt_pos, keras_shape))
        return

    cls_token = pt_pos[:, 0:1, :]
    grid_tokens = pt_pos[:, 1:, :]

    n_pt = grid_tokens.shape[1]
    gs_pt = int(np.sqrt(n_pt))
    n_keras = keras_shape[0] - 1
    gs_keras = int(np.sqrt(n_keras))

    grid_tokens = grid_tokens.reshape(1, gs_pt, gs_pt, -1)

    # Use PyTorch bicubic interpolation with size= to match exact DINOv2
    # runtime behaviour (align_corners=False, antialias=True).
    pt_tensor = torch.tensor(grid_tokens).permute(0, 3, 1, 2).to(dtype=torch.float32)  # fmt: skip
    grid_resized = torch.nn.functional.interpolate(
        pt_tensor,
        size=(gs_keras, gs_keras),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    grid_resized = grid_resized.permute(0, 2, 3, 1).numpy()
    grid_resized = grid_resized.reshape(1, -1, pt_pos.shape[-1])

    new_pos = np.concatenate([cls_token, grid_resized], axis=1)
    keras_pos.assign(np.reshape(new_pos, keras_shape))


def _transfer_full_backbone_weights(pt_backbone, keras_backbone):
    pt_encoder = pt_backbone.encoder.encoder
    k_model = keras_backbone.get_layer("encoder")

    # Transfer patch embeddings manually so we can interpolate pos embeds
    # when PT and Keras have different image_size (and thus different
    # position_embeddings shapes).
    from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (  # fmt: skip
        transfer_conv2d,
        to_keras,
        assign_table,
    )

    conv = k_model.get_layer("embeddings_patch_embeddings_projection")
    transfer_conv2d(pt_encoder.embeddings.patch_embeddings.projection, conv)
    cls = k_model.get_layer("embeddings_cls_token").embeddings
    assign_table(cls, to_keras(pt_encoder.embeddings.cls_token))
    pos = k_model.get_layer("embeddings_position_embeddings").embeddings
    _resize_and_assign_pos_embed(pt_encoder.embeddings, pos)
    registers = optional_embedding_table(k_model, "embeddings_register_tokens")
    if pt_encoder.embeddings.register_tokens is not None and registers is not None:  # fmt: skip
        assign_table(registers, to_keras(pt_encoder.embeddings.register_tokens))

    transfer_encoder(pt_encoder.encoder, k_model, "encoder")
    transfer_layernorm(pt_encoder.layernorm, k_model.get_layer("layernorm"))

    projector = keras_backbone.get_layer("projector")
    port_weights_multiscale_projector(pt_backbone.projector, projector)


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_backbone_real_weights_parity(variant):
    cfg = MODEL_CONFIGS[variant]
    res = cfg["resolution"]

    print(f"\\n{'='*60}")
    print(f"Testing Backbone parity for RFDETR {variant} (res={res})")
    print(f"{'='*60}")

    print(f"Loading pretrained {cfg['cls'].__name__}...")
    pt_backbone, _, _ = _extract_pt_parts(cfg["cls"])
    pt_backbone = pt_backbone.cpu()

    print("Building Keras Backbone...")
    keras_backbone = _build_keras_backbone(cfg)

    print("Transferring weights...")
    _transfer_full_backbone_weights(pt_backbone, keras_backbone)

    np.random.seed(42)
    x_np = np.random.randn(1, res, res, 3).astype(np.float32) * 0.1
    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    mask_np = np.zeros((1, res, res), dtype=bool)
    mask_pt = torch.from_numpy(mask_np)

    print("Running forward passes...")
    with torch.no_grad():
        from rfdetr.util.misc import NestedTensor

        nested = NestedTensor(x_pt, mask_pt)
        pt_outs = pt_backbone(nested)

    k_outs = keras_backbone([x_np, mask_np], training=False)

    assert len(pt_outs) == len(k_outs)

    for i, (pt_out, k_out) in enumerate(zip(pt_outs, k_outs)):
        pt_feat = pt_out.tensors.detach().cpu().numpy()
        pt_feat = chw_to_hwc(pt_feat)

        k_feat, k_mask = k_out
        k_feat = np.array(k_feat)

        print(f"  Scale {i}: PT shape={pt_feat.shape}, Keras shape={k_feat.shape}")  # fmt: skip

        assert pt_feat.shape == k_feat.shape

        diff = np.abs(k_feat - pt_feat)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")

        err = f"Scale {i} feature mismatch"
        np.testing.assert_allclose(
            k_feat, pt_feat, atol=1e-4, rtol=1e-4, err_msg=err
        )

    print(f"RFDETR {variant} Backbone parity PASSED")


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_backbone_output_shapes_real_weights(variant):
    cfg = MODEL_CONFIGS[variant]
    res = cfg["resolution"]

    keras_backbone = _build_keras_backbone(cfg)

    x_np = np.zeros((1, res, res, 3), dtype=np.float32)
    mask_np = np.zeros((1, res, res), dtype=bool)

    outs = keras_backbone([x_np, mask_np], training=False)

    # Expect 1 scale (stride 16 for P4)
    strides = [16]

    assert len(outs) == 1
    for i, (feat, mask) in enumerate(outs):
        h, w = res // strides[i], res // strides[i]
        feat_msg = f"Scale {i} shape mismatch: {feat.shape}"
        assert feat.shape == (1, h, w, 256), feat_msg
        mask_msg = f"Scale {i} mask shape mismatch: {mask.shape}"
        assert mask.shape == (1, h, w), mask_msg

    print(f"Output shapes correct for RFDETR {variant}")


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_backbone_mask_handling(variant):
    cfg = MODEL_CONFIGS[variant]
    res = cfg["resolution"]

    pt_backbone, _, _ = _extract_pt_parts(cfg["cls"])
    pt_backbone = pt_backbone.cpu()

    keras_backbone = _build_keras_backbone(cfg)
    _transfer_full_backbone_weights(pt_backbone, keras_backbone)

    dummy_img = np.zeros((1, res, res, 3), dtype=np.float32)
    dummy_mask = np.zeros((1, res, res), dtype=bool)
    outs = keras_backbone([dummy_img, dummy_mask], training=False)

    for i, (feat, mask) in enumerate(outs):
        msg = f"Scale {i} mask should be all False (unmasked)"
        assert not np.any(mask), msg

    true_mask = np.ones((1, res, res), dtype=bool)
    outs_masked = keras_backbone([dummy_img, true_mask], training=False)
    for i, (feat, mask) in enumerate(outs_masked):
        assert np.all(mask), f"Scale {i} mask should be all True (masked)"

    print(f"Mask handling correct for RFDETR {variant}")


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_joiner_real_weights_parity(variant):
    cfg = MODEL_CONFIGS[variant]
    res = cfg["resolution"]

    print(f"\\n{'='*60}")
    print(f"Testing Joiner parity for RFDETR {variant} (res={res})")
    print(f"{'='*60}")

    print(f"Loading pretrained {cfg['cls'].__name__}...")
    pt_backbone, pt_pos_embed, inner = _extract_pt_parts(cfg["cls"])
    pt_backbone = pt_backbone.cpu()
    pt_joiner = inner.backbone.cpu()

    print("Building Keras Joiner...")
    keras_joiner = _build_keras_joiner(cfg)
    keras_backbone = keras_joiner.get_layer("backbone")

    print("Transferring weights...")
    _transfer_full_backbone_weights(pt_backbone, keras_backbone)

    np.random.seed(99)
    x_np = np.random.randn(1, res, res, 3).astype(np.float32) * 0.1
    mask_np = np.zeros((1, res, res), dtype=bool)

    x_pt = torch.from_numpy(hwc_to_chw(x_np))
    mask_pt = torch.from_numpy(mask_np)

    from rfdetr.util.misc import NestedTensor

    nested = NestedTensor(x_pt, mask_pt)

    print("Running forward passes...")
    with torch.no_grad():
        pt_outs, pt_pos = pt_joiner(nested)

    k_outs, k_pos = keras_joiner([x_np, mask_np], training=False)

    print("Comparing features...")
    for i, (pt_out, k_out) in enumerate(zip(pt_outs, k_outs)):
        pt_feat = chw_to_hwc(pt_out.tensors.detach().cpu().numpy())
        k_feat_arr = np.array(k_out[0])

        assert pt_feat.shape == k_feat_arr.shape
        np.testing.assert_allclose(k_feat_arr, pt_feat, atol=1e-4, rtol=1e-4)

    print("Comparing position embeddings...")
    for i, (pt_p, k_p) in enumerate(zip(pt_pos, k_pos)):
        pt_p_np = chw_to_hwc(pt_p.detach().cpu().numpy())
        k_p_np = np.array(k_p)

        print(f"  Pos {i}: PT={pt_p_np.shape}, Keras={k_p_np.shape}")
        assert pt_p_np.shape == k_p_np.shape
        # Sine encoding should be identical if logic is same
        np.testing.assert_allclose(k_p_np, pt_p_np, atol=1e-4, rtol=1e-4)

    print(f"RFDETR {variant} Joiner parity PASSED")


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_build_backbone_factory(variant):
    cfg = MODEL_CONFIGS[variant]
    res = cfg["resolution"]

    print(f"\\nTesting build_backbone factory for RFDETR {variant}")

    joiner = _build_keras_joiner(cfg)

    assert isinstance(joiner, keras.Model)
    assert joiner.name == "joiner"
    assert joiner.get_layer("backbone").name == "backbone"

    x_np = np.random.randn(1, res, res, 3).astype(np.float32) * 0.1
    mask_np = np.zeros((1, res, res), dtype=bool)
    x_out, pos_out = joiner([x_np, mask_np], training=False)

    assert len(x_out) == 1
    assert len(pos_out) == 1
    print(f"Factory build successful for {variant}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
