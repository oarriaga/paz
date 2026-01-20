import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn
import keras
from unittest.mock import patch

# -------------------------------------------------------------------------
# 0. Environment Setup
# -------------------------------------------------------------------------
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["bat"] = "1"

torch.set_num_threads(1)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# Add project root to path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from examples.dino_object_detection.models.utils.get_param_dicts import (
    get_param_dict as get_keras_params,
    get_vit_lr_decay_rate as keras_decay_rate,
    get_weight_decay_rate as keras_wd_rate,
)
from examples.dino_object_detection.models.utils.torch_get_params_dicts_for_testing import (
    get_param_dict as get_torch_params,
    get_vit_lr_decay_rate as torch_decay_rate,
    get_vit_weight_decay_rate as torch_wd_rate,
)

# -------------------------------------------------------------------------
# MOCKS & HELPERS
# -------------------------------------------------------------------------


class MockConfig:
    def __init__(self):
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.lr_vit_layer_decay = 0.75
        self.weight_decay = 1e-4
        self.lr_component_decay = 0.5
        self.num_layers = 12


class MockTorchJoiner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(12)])
        self.pos_embed = nn.Parameter(torch.randn(1, 1))
        self.cls_token = nn.Parameter(torch.randn(1, 1))

    def __getitem__(self, idx):
        # Allow Joiner[0] to work as expected by torch_get_params_dicts
        return self

    def get_named_param_lr_pairs(self, args, prefix):
        pairs = {}
        for name, p in self.named_parameters():
            full_name = f"{prefix}.{name}"
            lr_scale = torch_decay_rate(
                full_name, args.lr_vit_layer_decay, args.num_layers
            )
            wd = torch_wd_rate(full_name, args.weight_decay)
            pairs[full_name] = {
                "params": p,
                "lr": args.lr_backbone * lr_scale,
                "weight_decay": wd,
            }
        return pairs


class MockTorchModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Direct assignment to simulate the unwrapped Joiner structure
        self.backbone = MockTorchJoiner(args)
        self.transformer = nn.ModuleDict({"decoder": nn.Linear(1, 1)})
        self.input_proj = nn.Linear(1, 1)


class MockKerasBackbone(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Naming ensures regex matches: blocks_0, blocks_1, etc.
        self.blocks = [keras.layers.Dense(1, name=f"blocks_{i}") for i in range(12)]
        self.pos_embed = self.add_weight(name="pos_embed", shape=(1, 1))
        self.cls_token = self.add_weight(name="cls_token", shape=(1, 1))

    def build(self, input_shape):
        for block in self.blocks:
            block.build(input_shape)
        super().build(input_shape)


class MockKerasJoiner(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = MockKerasBackbone(name="dinov2_encoder")

    def build(self, input_shape):
        self.backbone.build(input_shape)
        super().build(input_shape)


class MockKerasModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.joiner = MockKerasJoiner(name="joiner")
        # Naming ensures component decay logic triggers
        self.transformer_decoder = keras.layers.Dense(1, name="transformer_decoder_0")
        self.input_proj = keras.layers.Dense(1, name="input_proj")

    def build(self, input_shape):
        self.joiner.build(input_shape)
        self.transformer_decoder.build(input_shape)
        self.input_proj.build(input_shape)
        super().build(input_shape)


# -------------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------------


def test_lr_decay_calculation_match():
    cases = [
        ("backbone.0.pos_embed", "dinov2_encoder/pos_embed"),
        ("backbone.0.cls_token", "dinov2_encoder/cls_token"),
        # Verifying equivalent paths
        ("backbone.0.blocks.0.weight", "dinov2_encoder/blocks_0/kernel"),
        ("backbone.0.blocks.5.weight", "dinov2_encoder/blocks_5/kernel"),
        ("backbone.0.blocks.11.weight", "dinov2_encoder/blocks_11/kernel"),
    ]

    decay = 0.75
    layers = 12

    for torch_name, keras_name in cases:
        t_rate = torch_decay_rate(torch_name, decay, layers)
        k_rate = keras_decay_rate(keras_name, decay, layers)
        assert np.isclose(t_rate, k_rate), f"Mismatch {torch_name} vs {keras_name}"


def test_weight_decay_exclusion_match():
    cases = [
        ("backbone.0.norm.weight", "norm/gamma", "gamma", 0.0),
        ("backbone.0.blocks.0.bias", "blocks_0/bias", "bias", 0.0),
        ("backbone.0.pos_embed", "pos_embed", "pos_embed", 0.0),
        ("backbone.0.blocks.0.weight", "blocks_0/kernel", "kernel", 1e-4),
    ]

    wd_val = 1e-4

    for t_name, k_path, k_name, expected in cases:
        mock_k_var = type("obj", (object,), {"path": k_path, "name": k_name})
        t_wd = torch_wd_rate(t_name, wd_val)
        k_wd = keras_wd_rate(mock_k_var, wd_val)
        assert t_wd == expected, f"Torch failed {t_name}"
        assert k_wd == expected, f"Keras failed {k_name}"


def test_full_optimizer_dict_structure():
    args = MockConfig()
    t_model = MockTorchModel(args)
    k_model = MockKerasModel()
    k_model.build((1, 1))

    # Patch Joiner to pass isinstance check in BOTH torch and keras implementations
    with (
        patch(
            "examples.dino_object_detection.models.utils.torch_get_params_dicts_for_testing.Joiner",
            MockTorchJoiner,
        ),
        patch(
            "examples.dino_object_detection.models.utils.get_param_dicts.Joiner",
            MockKerasJoiner,
        ),
    ):
        t_dicts = get_torch_params(args, t_model)
        k_dicts = get_keras_params(args, k_model)

    assert len(t_dicts) > 0
    assert len(k_dicts) > 0

    t_lrs = sorted([d["lr"] for d in t_dicts])
    k_lrs = sorted([d["lr"] for d in k_dicts])

    t_unique = set(round(lr, 8) for lr in t_lrs)
    k_unique = set(round(lr, 8) for lr in k_lrs)

    assert (
        t_unique == k_unique
    ), f"LR Sets Mismatch!\nTorch: {t_unique}\nKeras: {k_unique}"

    # Check "Other" group specifically (input_proj)
    t_input_proj_param = t_model.input_proj.weight
    t_group = next(d for d in t_dicts if d["params"] is t_input_proj_param)

    t_wd = t_group.get("weight_decay", args.weight_decay)

    assert t_group["lr"] == args.lr
    assert t_wd == args.weight_decay, "Torch implicit WD mismatch"

    # Keras
    k_group = next(d for d in k_dicts if "input_proj" in d["name"])
    assert k_group["lr"] == args.lr
    assert k_group["weight_decay"] == args.weight_decay, "Keras explicit WD mismatch"

    # Verify parity
    assert t_wd == k_group["weight_decay"], "Parity Mismatch on Weight Decay"


def test_decoder_component_decay():
    args = MockConfig()
    t_model = MockTorchModel(args)
    k_model = MockKerasModel()
    k_model.build((1, 1))

    with (
        patch(
            "examples.dino_object_detection.models.utils.torch_get_params_dicts_for_testing.Joiner",
            MockTorchJoiner,
        ),
        patch(
            "examples.dino_object_detection.models.utils.get_param_dicts.Joiner",
            MockKerasJoiner,
        ),
    ):
        t_dicts = get_torch_params(args, t_model)
        k_dicts = get_keras_params(args, k_model)

    expected_lr = args.lr * args.lr_component_decay

    decoder_weight = t_model.transformer["decoder"].weight
    t_decoder_group = next(d for d in t_dicts if d["params"] is decoder_weight)

    k_decoder_group = next(d for d in k_dicts if "decoder" in d["name"])

    assert t_decoder_group["lr"] == expected_lr
    assert k_decoder_group["lr"] == expected_lr


if __name__ == "__main__":
    pytest.main([__file__])
