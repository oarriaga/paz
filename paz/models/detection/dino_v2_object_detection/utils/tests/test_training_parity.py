import math
import os
import sys
import importlib.util

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import Keras utilities under test
# ---------------------------------------------------------------------------

_UTILS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _UTILS_DIR)

from get_param_dicts import (
    get_vit_lr_decay_rate,
    get_vit_weight_decay_rate,
    classify_variable,
    compute_backbone_lr,
    build_lr_scale_map,
    scale_gradients_by_lr,
)

# Import ModelEma via explicit path (avoid stdlib ``utils`` shadow)
_utils_path = os.path.join(_UTILS_DIR, "utils.py")
_spec = importlib.util.spec_from_file_location("keras_utils", _utils_path)
_keras_utils = importlib.util.module_from_spec(_spec)
sys.modules["keras_utils"] = _keras_utils
_spec.loader.exec_module(_keras_utils)
ModelEma = _keras_utils.ModelEma

# Import engine helpers
_ENGINE_DIR = os.path.dirname(_UTILS_DIR)
sys.path.insert(0, _ENGINE_DIR)
from engine import build_lr_lambda, _clip_grad_norm


# ---------------------------------------------------------------------------
# PyTorch reference helpers (pure Python, no torch needed)
# ---------------------------------------------------------------------------


def _pytorch_dinov2_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """Reproduce ``get_dinov2_lr_decay_rate`` from PyTorch backbone.py."""
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if "embeddings" in name:
            layer_id = 0
        elif ".layer." in name and ".residual." not in name:
            layer_id = int(
                name[name.find(".layer."):].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def _pytorch_dinov2_weight_decay_rate(name, weight_decay_rate=1.0):
    """Reproduce ``get_dinov2_weight_decay_rate`` from PyTorch backbone.py."""
    keywords = ("gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings")
    if any(kw in name for kw in keywords):
        return 0.0
    return weight_decay_rate


def _pytorch_ema_get_decay(decay, tau, updates):
    """Reproduce ``ModelEma._get_decay`` from PyTorch utils.py."""
    if tau == 0:
        return decay
    return decay * (1 - math.exp(-updates / tau))


def _pytorch_backbone_lr(name, *, lr_encoder, lr_vit_layer_decay,
                          lr_component_decay, num_layers):
    """Reproduce LR formula from PyTorch ``Backbone.get_named_param_lr_pairs``.

    ``lr = lr_encoder × layer_decay(name) × lr_component_decay²``
    """
    layer_decay = _pytorch_dinov2_lr_decay_rate(
        name, lr_decay_rate=lr_vit_layer_decay, num_layers=num_layers)
    return lr_encoder * layer_decay * (lr_component_decay ** 2)


# ---------------------------------------------------------------------------
# 1. LR decay rate
# ---------------------------------------------------------------------------

_LR_DECAY_CASES = [
    # (param_name, num_layers, lr_decay_rate, description)
    ("backbone.0.encoder.embeddings.patch_embeddings.projection.weight",
     12, 0.8, "embeddings → layer 0"),
    ("backbone.0.encoder.layer.0.attention.attention.query.weight",
     12, 0.8, "ViT block 0 → layer 1"),
    ("backbone.0.encoder.layer.5.mlp.fc1.weight",
     12, 0.8, "ViT block 5 → layer 6"),
    ("backbone.0.encoder.layer.11.output.dense.weight",
     12, 0.8, "ViT block 11 (last) → layer 12"),
    ("transformer.decoder.layers.0.self_attn.in_proj_weight",
     12, 0.8, "decoder (non-backbone) → no decay"),
    ("class_embed.layers.0.kernel",
     12, 0.8, "head (non-backbone) → no decay"),
]


@pytest.mark.parametrize("name,num_layers,lr_decay,desc", _LR_DECAY_CASES,
                         ids=[c[3] for c in _LR_DECAY_CASES])
def test_lr_decay_rate_matches_pytorch(name, num_layers, lr_decay, desc):
    keras_rate = get_vit_lr_decay_rate(name, lr_decay, num_layers)
    pytorch_rate = _pytorch_dinov2_lr_decay_rate(name, lr_decay, num_layers)
    assert keras_rate == pytest.approx(pytorch_rate, abs=1e-10), (
        f"[{desc}] keras={keras_rate}, pytorch={pytorch_rate}")


# ---------------------------------------------------------------------------
# 2. Weight-decay exclusion
# ---------------------------------------------------------------------------

_WD_CASES = [
    ("backbone.0.encoder.layer.0.attention.attention.query.weight", 1.0),
    ("backbone.0.encoder.layer.0.attention.attention.query.bias", 0.0),
    ("backbone.0.encoder.layer.0.norm1.weight", 0.0),
    ("backbone.0.encoder.embeddings.patch_embeddings.projection.weight", 0.0),
    ("backbone.0.encoder.layer.0.attention.gamma", 0.0),
    ("transformer.decoder.layers.0.self_attn.in_proj_weight", 1.0),
    ("class_embed.layers.0.kernel", 1.0),
]


@pytest.mark.parametrize("name,expected_rate", _WD_CASES)
def test_weight_decay_rate_matches_pytorch(name, expected_rate):
    keras_wd = get_vit_weight_decay_rate(name)
    pytorch_wd = _pytorch_dinov2_weight_decay_rate(name)
    assert keras_wd == expected_rate
    assert keras_wd == pytorch_wd


# ---------------------------------------------------------------------------
# 3. Variable classification (backbone / decoder / other)
# ---------------------------------------------------------------------------

_CLASSIFY_CASES = [
    ("backbone/encoder/layer/0/kernel:0", "backbone"),
    ("backbone.0.encoder.embeddings.weight", "backbone"),
    ("transformer/decoder/layers/0/self_attn/kernel:0", "decoder"),
    ("transformer.decoder.layers.0.weight", "decoder"),
    ("class_embed/layers/0/kernel:0", "other"),
    ("bbox_embed/layers/0/kernel:0", "other"),
    ("query_embed/kernel:0", "other"),
]


@pytest.mark.parametrize("name,expected_group", _CLASSIFY_CASES)
def test_classify_variable(name, expected_group):
    assert classify_variable(name) == expected_group


# ---------------------------------------------------------------------------
# 4. Backbone LR formula
# ---------------------------------------------------------------------------

_BACKBONE_LR_CASES = [
    # (name, lr_encoder, lr_vit_layer_decay, lr_component_decay, num_layers)
    ("backbone.0.encoder.embeddings.weight", 1.5e-4, 0.8, 0.7, 12),
    ("backbone.0.encoder.layer.0.attention.weight", 1.5e-4, 0.8, 0.7, 12),
    ("backbone.0.encoder.layer.11.output.weight", 1.5e-4, 0.8, 0.7, 12),
    ("backbone.0.encoder.layer.5.mlp.weight", 1e-4, 0.9, 0.5, 12),
]


@pytest.mark.parametrize("name,lr_enc,decay,comp,nl", _BACKBONE_LR_CASES)
def test_backbone_lr_matches_pytorch(name, lr_enc, decay, comp, nl):
    keras_lr = compute_backbone_lr(
        name, lr_encoder=lr_enc, lr_vit_layer_decay=decay,
        lr_component_decay=comp, num_layers=nl)
    pytorch_lr = _pytorch_backbone_lr(
        name, lr_encoder=lr_enc, lr_vit_layer_decay=decay,
        lr_component_decay=comp, num_layers=nl)
    assert keras_lr == pytest.approx(pytorch_lr, rel=1e-10)


# ---------------------------------------------------------------------------
# 5. EMA decay schedule
# ---------------------------------------------------------------------------

_EMA_CASES = [
    # (decay, tau, updates, description)
    (0.993, 0, 1, "tau=0 → constant decay"),
    (0.993, 0, 100, "tau=0 → constant at step 100"),
    (0.993, 100, 1, "tau=100 → ramp-up step 1"),
    (0.993, 100, 50, "tau=100 → ramp-up step 50"),
    (0.993, 100, 100, "tau=100 → ramp-up step 100"),
    (0.993, 100, 500, "tau=100 → near-plateau step 500"),
]


@pytest.mark.parametrize("decay,tau,updates,desc", _EMA_CASES,
                         ids=[c[3] for c in _EMA_CASES])
def test_ema_decay_matches_pytorch(decay, tau, updates, desc):
    expected = _pytorch_ema_get_decay(decay, tau, updates)

    # Simulate Keras ModelEma internal state
    class FakeModel:
        def get_weights(self):
            return [np.zeros(1)]
        weights = [type("W", (), {
            "path": "fake/weight",
            "numpy": lambda self: np.zeros(1),
        })()]

    ema = ModelEma(FakeModel(), decay=decay, tau=tau)
    ema.updates = updates
    keras_decay = ema._get_decay()

    assert keras_decay == pytest.approx(expected, abs=1e-12), (
        f"[{desc}] keras={keras_decay}, pytorch={expected}")


# ---------------------------------------------------------------------------
# 6. LR schedule (warmup + cosine)
# ---------------------------------------------------------------------------


def _pytorch_lr_lambda_cosine(step, warmup_steps, total_steps):
    """Reproduce the cosine LR lambda from PyTorch main.py."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(
        max(1, total_steps - warmup_steps))
    return 0.5 * (1 + math.cos(math.pi * progress))


@pytest.mark.parametrize("steps_per_epoch,epochs,warmup_epochs", [
    (100, 50, 1),
    (50, 100, 2),
    (200, 30, 0.5),
])
def test_lr_schedule_matches_pytorch(steps_per_epoch, epochs, warmup_epochs):
    lr_lambda = build_lr_lambda(
        num_training_steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        lr_scheduler="cosine",
    )
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(steps_per_epoch * warmup_epochs)

    # Test at key checkpoints
    test_steps = [0, 1, warmup_steps // 2, warmup_steps, warmup_steps + 1,
                  total_steps // 2, total_steps - 1]
    for step in test_steps:
        keras_val = lr_lambda(step)
        pytorch_val = _pytorch_lr_lambda_cosine(step, warmup_steps, total_steps)
        assert keras_val == pytest.approx(pytorch_val, abs=1e-10), (
            f"Mismatch at step {step}: keras={keras_val}, pytorch={pytorch_val}")


# ---------------------------------------------------------------------------
# 7. Weight dict keys (aux loss — no cascading)
# ---------------------------------------------------------------------------


def _pytorch_weight_dict_keys(dec_layers, two_stage):
    """Build the expected set of weight-dict keys per PyTorch lwdetr.py."""
    base = {"loss_ce", "loss_bbox", "loss_giou"}
    result = set(base)
    for i in range(dec_layers - 1):
        result.update({f"{k}_{i}" for k in base})
    if two_stage:
        result.update({f"{k}_enc" for k in base})
    return result


def test_weight_dict_no_cascading():
    """Weight dict should have exactly (base × dec_layers) + encoder keys.

    This reproduces the weight_dict construction logic from main.py
    """
    # --- Reproduce the Keras weight_dict construction (from main.py) ---
    dec_layers = 3
    two_stage = True
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }
    base_weight_keys = list(weight_dict.items())
    for i in range(dec_layers - 1):
        weight_dict.update({k + f"_{i}": v for k, v in base_weight_keys})
    if two_stage:
        weight_dict.update({k + "_enc": v for k, v in base_weight_keys})

    keras_keys = set(weight_dict.keys())
    pytorch_keys = _pytorch_weight_dict_keys(dec_layers, two_stage)
    assert keras_keys == pytorch_keys, (
        f"Extra: {keras_keys - pytorch_keys}, "
        f"Missing: {pytorch_keys - keras_keys}")

    # Also verify no cascading: each aux key should NOT have double suffixes
    for key in keras_keys:
        # e.g. "loss_ce_0_1" would indicate cascading
        suffixes = key.replace("loss_ce", "").replace("loss_bbox", "").replace(
            "loss_giou", "")
        assert suffixes.count("_") <= 1, f"Cascading detected in key: {key}"


# ---------------------------------------------------------------------------
# 8. Gradient clipping (global norm)
# ---------------------------------------------------------------------------


def test_gradient_clipping_global_norm():
    """``_clip_grad_norm`` should reproduce PyTorch clip_grad_norm_ behaviour."""
    import keras
    from keras import ops

    # Create fake gradients with known norm
    g1 = ops.convert_to_tensor(
        np.array([3.0, 4.0], dtype="float32"))  # norm = 5
    g2 = ops.convert_to_tensor(
        np.array([0.0, 0.0], dtype="float32"))  # norm = 0
    total_norm = 5.0  # sqrt(9 + 16 + 0 + 0)

    # Clip to max_norm=2.5 → scale = 2.5 / 5.0 = 0.5
    clipped = _clip_grad_norm([g1, g2], max_norm=2.5)
    c1 = ops.convert_to_numpy(clipped[0])
    c2 = ops.convert_to_numpy(clipped[1])

    np.testing.assert_allclose(c1, [1.5, 2.0], atol=1e-5)
    np.testing.assert_allclose(c2, [0.0, 0.0], atol=1e-5)


def test_gradient_clipping_no_clip_when_small():
    """Gradients below max_norm should not be modified."""
    import keras
    from keras import ops

    g1 = ops.convert_to_tensor(
        np.array([0.01, 0.02], dtype="float32"))
    clipped = _clip_grad_norm([g1], max_norm=1.0)
    c1 = ops.convert_to_numpy(clipped[0])
    np.testing.assert_allclose(c1, [0.01, 0.02], atol=1e-6)


# ---------------------------------------------------------------------------
# 9. Per-component LR gradient scaling
# ---------------------------------------------------------------------------


def test_scale_gradients_by_lr():
    """Gradient scaling should multiply each grad by its lr_scale."""
    import keras

    class FakeVar:
        def __init__(self, name):
            self.name = name

    vars_ = [FakeVar("backbone/encoder/layer/0/kernel:0"),
             FakeVar("transformer/decoder/layers/0/kernel:0"),
             FakeVar("class_embed/layers/0/kernel:0")]

    lr_scale_map = {
        "backbone/encoder/layer/0/kernel:0": {"lr_scale": 0.5, "wd": 0.0},
        "transformer/decoder/layers/0/kernel:0": {"lr_scale": 0.7, "wd": 1e-4},
        "class_embed/layers/0/kernel:0": {"lr_scale": 1.0, "wd": 1e-4},
    }

    from keras import ops as _ops
    grads = [
        _ops.convert_to_tensor(np.ones(3, dtype="float32")),
        _ops.convert_to_tensor(np.ones(3, dtype="float32")),
        _ops.convert_to_tensor(np.ones(3, dtype="float32")),
    ]

    scaled = scale_gradients_by_lr(grads, vars_, lr_scale_map)
    from keras import ops
    np.testing.assert_allclose(
        ops.convert_to_numpy(scaled[0]), [0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(
        ops.convert_to_numpy(scaled[1]), [0.7, 0.7, 0.7], atol=1e-6)
    np.testing.assert_allclose(
        ops.convert_to_numpy(scaled[2]), [1.0, 1.0, 1.0], atol=1e-6)


def test_scale_gradients_handles_none():
    """None gradients (frozen params) should pass through unchanged."""

    class FakeVar:
        def __init__(self, name):
            self.name = name

    vars_ = [FakeVar("backbone/kernel:0")]
    lr_scale_map = {"backbone/kernel:0": {"lr_scale": 0.5, "wd": 0.0}}

    scaled = scale_gradients_by_lr([None], vars_, lr_scale_map)
    assert scaled[0] is None
