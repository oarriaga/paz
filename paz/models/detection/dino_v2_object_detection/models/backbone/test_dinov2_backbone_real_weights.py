import numpy as np
import pytest
import torch
import sys
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

rfdetr_parent = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../../../../../../",
        "examples", "rf-detr_original_pytorch_implementation",
    )
)
if rfdetr_parent not in sys.path:
    sys.path.insert(0, rfdetr_parent)

from paz.models.detection.dino_v2_object_detection.models.backbone.dinov2 import (  # fmt: skip
    DinoV2,
)

from paz.models.detection.dino_v2_object_detection.models.backbone.backbone_weights_porting_utils import (  # fmt: skip
    transfer_patch_embeddings,
    transfer_encoder,
    transfer_layernorm,
    chw_to_hwc,
    hwc_to_chw,
)

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
except ImportError:
    sys.path.append(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../../../../../../",
                "examples", "rf-detr_original_pytorch_implementation",
            )
        )
    )
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge


# NOTE: PyTorch configs use 1-based indexing (3, 6, 9, 12).
# Keras implementation uses 0-based indexing (0..11).
# We map [3, 6, 9, 12] -> [2, 5, 8, 11].
MODEL_CONFIGS = {
    "Nano": {
        "cls": RFDETRNano,
        "out_feature_indexes": [2, 5, 8, 11],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 384,
        "positional_encoding_size": 24,
    },
    "Small": {
        "cls": RFDETRSmall,
        "out_feature_indexes": [2, 5, 8, 11],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 512,
        "positional_encoding_size": 32,
    },
    "Medium": {
        "cls": RFDETRMedium,
        "out_feature_indexes": [2, 5, 8, 11],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 576,
        "positional_encoding_size": 36,
    },
    "Large": {
        "cls": RFDETRLarge,
        "out_feature_indexes": [2, 5, 8, 11],
        "window_block_indexes": [0, 1, 2, 4, 5, 7, 8, 10, 11],
        "patch_size": 16,
        "num_windows": 2,
        "resolution": 704,
        "positional_encoding_size": 704 // 16,
    },
}


def _extract_pt_dinov2(model_class):
    torch_full_model = model_class(pretrained=True)
    inner = torch_full_model.model.model
    inner.eval()

    backbone = inner.backbone
    # Handle Joiner wrapper: backbone[0] is the actual Backbone
    if hasattr(backbone, "__getitem__"):
        try:
            base_backbone = backbone[0]
        except Exception:
            base_backbone = backbone
    else:
        base_backbone = backbone

    assert hasattr(base_backbone, "encoder"), (
        f"Could not find .encoder on backbone of {model_class.__name__}"
    )
    pt_dinov2 = base_backbone.encoder  # PyTorch DinoV2 wrapper
    pt_dinov2.eval()
    return pt_dinov2


def _transfer_dinov2_weights(pt_dinov2, keras_dinov2):
    pt_encoder = pt_dinov2.encoder
    k_model = keras_dinov2

    transfer_patch_embeddings(pt_encoder.embeddings, k_model, "embeddings")

    transfer_encoder(pt_encoder.encoder, k_model, "encoder")

    transfer_layernorm(pt_encoder.layernorm, k_model.get_layer("layernorm"))


def _build_keras_dinov2(cfg):
    return DinoV2(
        shape=(cfg["resolution"], cfg["resolution"]),
        out_feature_indexes=cfg["out_feature_indexes"],
        size="small",  # All variants use dinov2_windowed_small
        # encoder name has no "registers" -> no register tokens
        use_registers=False,
        use_windowed_attn=True,
        patch_size=cfg["patch_size"],
        num_windows=cfg["num_windows"],
        window_block_indexes=cfg["window_block_indexes"],
        positional_encoding_size=cfg["positional_encoding_size"],
    )


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_dinov2_encoder_real_weights_parity(variant):
    cfg = MODEL_CONFIGS[variant]
    model_class = cfg["cls"]
    print(f"\\n{'='*60}")
    print(f"Testing DinoV2 encoder parity for RFDETR {variant}")
    print(f"{'='*60}")

    print(f"Loading pretrained {model_class.__name__}...")
    pt_dinov2 = _extract_pt_dinov2(model_class)
    pt_dinov2 = pt_dinov2.cpu()

    print("Building Keras DinoV2...")
    keras_dinov2 = _build_keras_dinov2(cfg)

    res = cfg["resolution"]
    dummy = np.zeros((1, res, res, 3), dtype=np.float32)
    _ = keras_dinov2(dummy, training=False)

    print("Transferring weights...")
    _transfer_dinov2_weights(pt_dinov2, keras_dinov2)

    np.random.seed(42)
    x_np = np.random.randn(1, res, res, 3).astype(np.float32) * 0.1
    x_pt = torch.from_numpy(hwc_to_chw(x_np))

    print("Running forward passes...")
    with torch.no_grad():
        pt_outputs = pt_dinov2(x_pt)

    keras_outputs = keras_dinov2(x_np, training=False)

    assert len(pt_outputs) == len(keras_outputs), (
        f"Output count mismatch: PT={len(pt_outputs)}, "
        f"Keras={len(keras_outputs)}"
    )

    for i, (pt_out, k_out) in enumerate(zip(pt_outputs, keras_outputs)):
        pt_np = pt_out.detach().cpu().numpy()
        pt_np = chw_to_hwc(pt_np)
        k_np = np.array(k_out)

        assert pt_np.shape == k_np.shape, (
            f"Scale {i}: shape mismatch PT={pt_np.shape} vs Keras={k_np.shape}"
        )

        max_diff = np.max(np.abs(pt_np - k_np))
        mean_diff = np.mean(np.abs(pt_np - k_np))
        print(
            f"  Scale {i}: shape={k_np.shape}, "
            f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
        )

        np.testing.assert_allclose(
            k_np, pt_np, atol=1e-4, rtol=1e-4,
            err_msg=f"Scale {i} output mismatch for RFDETR {variant}",
        )

    print(f"✓ RFDETR {variant} DinoV2 encoder parity PASSED")


@pytest.mark.parametrize("variant", list(MODEL_CONFIGS.keys()))
def test_dinov2_output_shapes_real_weights(variant):
    cfg = MODEL_CONFIGS[variant]
    model_class = cfg["cls"]
    res = cfg["resolution"]
    ps = cfg["patch_size"]
    expected_spatial = res // ps

    msg = f"\\nChecking output shapes for RFDETR {variant} (res={res}, ps={ps})"
    print(msg)

    pt_dinov2 = _extract_pt_dinov2(model_class)
    pt_dinov2 = pt_dinov2.cpu()

    keras_dinov2 = _build_keras_dinov2(cfg)
    dummy = np.zeros((1, res, res, 3), dtype=np.float32)
    _ = keras_dinov2(dummy, training=False)
    _transfer_dinov2_weights(pt_dinov2, keras_dinov2)

    x_np = np.random.randn(1, res, res, 3).astype(np.float32) * 0.1
    outputs = keras_dinov2(x_np, training=False)

    num_expected = len(cfg["out_feature_indexes"])
    assert len(outputs) == num_expected, (
        f"Expected {num_expected} outputs, got {len(outputs)}"
    )

    for i, out in enumerate(outputs):
        shape = tuple(np.array(out).shape)
        assert shape == (1, expected_spatial, expected_spatial, 384), (
            f"Scale {i}: expected "
            f"(1, {expected_spatial}, {expected_spatial}, 384), "
            f"got {shape}"
        )

    print(f"✓ Output shapes correct for RFDETR {variant}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
