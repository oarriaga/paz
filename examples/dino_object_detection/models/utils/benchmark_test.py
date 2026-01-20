import sys
import os
import json
import shutil
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# -------------------------------------------------------------------------
# 0. Environment Setup (MUST be before import keras)
# -------------------------------------------------------------------------
# Critical for Windows JAX stability:
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["bat"] = "1"

# PyTorch Setup
torch.set_num_threads(1)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# Add project root to path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add current directory to path
sys.path.append(os.getcwd())

# -------------------------------------------------------------------------
# 1. Imports (Must happen AFTER environment setup)
# -------------------------------------------------------------------------
import keras  # Import Keras only after setting JAX_PLATFORMS and KERAS_BACKEND

try:
    import examples.dino_object_detection.models.utils.torch_benchmark_for_testing as torch_bench
    import examples.dino_object_detection.models.utils.benchmark as keras_bench
except ImportError:
    pytest.fail(
        "Could not import benchmark files. Ensure 'torch_benchmark_for_testing.py' and 'benchmark.py' are in the current directory."
    )

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def assert_flops_near_equal(torch_flops, keras_flops, key_mapping=None, tolerance=1e-6):
    """
    Asserts that the FLOP counts for specific keys match between the two implementations.
    """
    if key_mapping is None:
        key_mapping = {k: k for k in torch_flops.keys()}

    for t_key, k_key in key_mapping.items():
        t_val = torch_flops.get(t_key, 0.0)
        k_val = keras_flops.get(k_key, 0.0)

        assert abs(t_val - k_val) <= tolerance, (
            f"FLOP mismatch for op '{t_key}' (Keras mapped: '{k_key}'). "
            f"Torch: {t_val}, Keras: {k_val}"
        )


def create_dummy_data_torch(shape, batch_size=1):
    return torch.randn(batch_size, *shape)


def create_dummy_data_keras(shape, batch_size=1):
    return keras.ops.ones((batch_size, *shape))


# --------------------------------------------------------------------------
# Layer-wise Logic Tests
# --------------------------------------------------------------------------
def test_nested_model_traversal_fixed():
    """Fixed version with correct GFLOPs expectation"""

    class Block(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(10)
            self.dense2 = keras.layers.Dense(10)

        def call(self, x):
            return self.dense2(self.dense1(x))

    inputs = keras.Input(shape=(10,))
    x = Block()(inputs)
    x = keras.layers.Dense(5)(x)
    k_model = keras.Model(inputs, x)

    k_input = create_dummy_data_keras((10,), batch_size=1)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # 250 FLOPs = 2.5e-07 GFLOPs
    expected_gflops = 250 / 1e9  # = 2.5e-07
    assert (
        abs(k_res["addmm"] - expected_gflops) < 1e-10
    ), f"Expected {expected_gflops} GFLOPs (250 FLOPs), got {k_res.get('addmm')}"


def test_multihead_attention_flops_fixed():
    """Fixed version"""
    B, S, EmbedDim = 1, 64, 256
    NumHeads = 4

    # PyTorch
    t_model = nn.MultiheadAttention(embed_dim=EmbedDim, num_heads=NumHeads, bias=True)
    t_model.batch_first = True
    t_input = create_dummy_data_torch((S, EmbedDim), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input, t_input, t_input))
    t_total = sum(t_res.values())

    # Keras - pass MHA layer directly
    k_model = keras.layers.MultiHeadAttention(
        num_heads=NumHeads, key_dim=EmbedDim // NumHeads
    )
    k_model.build(query_shape=(B, S, EmbedDim), value_shape=(B, S, EmbedDim))
    k_input = create_dummy_data_keras((S, EmbedDim), batch_size=B)

    k_res = keras_bench.flop_count(k_model, ([k_input, k_input],))
    k_total = sum(k_res.values())

    print(f"\nMHA Total FLOPs -> Torch: {t_total} | Keras: {k_total}")
    assert (
        abs(t_total - k_total) <= 1e-3
    ), f"MHA Total FLOP mismatch. T: {t_total}, K: {k_total}"


def test_dense_linear_flops():
    B, In, Out = 2, 64, 32

    # PyTorch Setup
    t_model = nn.Linear(In, Out, bias=True)
    t_input = create_dummy_data_torch((In,), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    # Keras Setup
    k_model = keras.layers.Dense(Out, use_bias=True)
    k_model.build((B, In))
    k_input = create_dummy_data_keras((In,), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # MAPPING FIX: Torch uses "linear", Keras uses "addmm"
    assert_flops_near_equal(t_res, k_res, key_mapping={"linear": "addmm"})


def test_conv2d_flops():
    B, C_in, C_out = 1, 3, 16
    H, W = 32, 32
    K = 3

    t_model = nn.Conv2d(C_in, C_out, K, padding=1)
    t_input = create_dummy_data_torch((C_in, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.Conv2D(C_out, K, padding="same", data_format="channels_last")
    k_model.build((B, H, W, C_in))
    k_input = create_dummy_data_keras((H, W, C_in), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    assert_flops_near_equal(t_res, k_res, key_mapping={"conv": "conv"})


def test_depthwise_conv2d_flops():
    B, C_in = 1, 16
    H, W = 64, 64
    K = 3

    t_model = nn.Conv2d(C_in, C_in, K, groups=C_in, padding=1)
    t_input = create_dummy_data_torch((C_in, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.DepthwiseConv2D(
        K, padding="same", data_format="channels_last"
    )
    k_model.build((B, H, W, C_in))
    k_input = create_dummy_data_keras((H, W, C_in), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    assert_flops_near_equal(t_res, k_res, key_mapping={"conv": "conv"})


def test_batch_norm_flops():
    B, C = 2, 32
    H, W = 16, 16

    t_model = nn.BatchNorm2d(C, affine=True)
    t_input = create_dummy_data_torch((C, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.BatchNormalization(axis=-1, center=True, scale=True)
    k_model.build((B, H, W, C))
    k_input = create_dummy_data_keras((H, W, C), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    assert_flops_near_equal(t_res, k_res, key_mapping={"batchnorm": "batchnorm"})


def test_relu_flops():
    shape = (10, 10, 10)

    t_model = nn.ReLU()
    t_input = create_dummy_data_torch(shape, batch_size=1)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.ReLU()
    k_model.build((1, *shape))
    k_input = create_dummy_data_keras(shape, batch_size=1)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    assert_flops_near_equal(t_res, k_res, key_mapping={"aten::relu": "aten::relu"})


def test_dropout_flops():
    shape = (50, 50)

    t_model = nn.Dropout(0.5)
    t_input = create_dummy_data_torch(shape, batch_size=1)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.Dropout(0.5)
    k_model.build((1, *shape))
    k_input = create_dummy_data_keras(shape, batch_size=1)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    assert_flops_near_equal(t_res, k_res, key_mapping={"dropout": "dropout"})


def test_get_shape_logic():
    k_shape = keras_bench.get_shape((None, 32, 32, 3))
    assert k_shape == [1, 32, 32, 3]

    k_shape_2 = keras_bench.get_shape([2, 10])
    assert k_shape_2 == [2, 10]


class MockDataset:
    def __init__(self, length=25, shape=(3, 32, 32)):
        self.length = length
        self.shape = shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.randn(*self.shape), 0

    def __iter__(self):
        for _ in range(self.length):
            yield (
                np.random.randn(*self.shape).transpose(1, 2, 0).astype(np.float32),
                0,
            )


def test_layer_norm_flops():
    # Transformers use LayerNorm, which PyTorch counts differently than BatchNorm
    B, C, H, W = 2, 32, 16, 16

    # PyTorch: affine=True usually results in 5 ops per element
    t_model = nn.LayerNorm([C, H, W], elementwise_affine=True)
    t_input = create_dummy_data_torch((C, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    # Keras
    k_model = keras.layers.LayerNormalization(axis=[1, 2, 3], scale=True, center=True)
    k_model.build((B, H, W, C))  # Keras usually channels last, but LN flexible
    k_input = create_dummy_data_keras((H, W, C), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # Map 'layer_norm' or 'norm' to Keras 'norm'
    # Note: Check if your torch benchmark outputs "layer_norm" or just "norm"
    t_key = "layer_norm" if "layer_norm" in t_res else "norm"
    assert_flops_near_equal(t_res, k_res, key_mapping={t_key: "norm"})


def test_elementwise_add_flops():
    # Residual connections (Skip connections)
    shape = (10, 10, 32)

    class TorchAdd(nn.Module):
        def forward(self, x):
            return x + x

    t_model = TorchAdd()
    t_input = create_dummy_data_torch(shape, batch_size=1)
    # PyTorch benchmark specifically looks for 'aten::add'
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.Add()
    k_input = create_dummy_data_keras(shape, batch_size=1)
    # Pass list of inputs for Add layer
    k_model.build([(1, *shape), (1, *shape)])
    k_res = keras_bench.flop_count(k_model, ([k_input, k_input],))

    assert_flops_near_equal(t_res, k_res, key_mapping={"aten::add": "elementwise"})


def test_softmax_flops():
    shape = (10, 100)

    t_model = nn.Softmax(dim=-1)
    t_input = create_dummy_data_torch(shape, batch_size=1)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.Softmax()
    k_model.build((1, *shape))
    k_input = create_dummy_data_keras(shape, batch_size=1)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # FIX: Map 'softmax' (key returned by PyTorch handler) to 'softmax'
    assert_flops_near_equal(t_res, k_res, key_mapping={"softmax": "softmax"})


# --------------------------------------------------------------------------
# Missing Critical Tests
# --------------------------------------------------------------------------


def test_multihead_attention_flops():
    # Transformers are the hardest to align.
    # PyTorch breaks MHA down into linear+bmm. Keras estimates it as one block.
    B, S, EmbedDim = 1, 64, 256
    NumHeads = 4

    # PyTorch
    t_model = nn.MultiheadAttention(embed_dim=EmbedDim, num_heads=NumHeads, bias=True)
    t_input = create_dummy_data_torch((S, EmbedDim), batch_size=B)  # (B, S, E)
    # PyTorch MHA expects (S, B, E) by default unless batch_first=True
    t_model.batch_first = True

    # Trace PyTorch Ops
    t_res = torch_bench.flop_count(t_model, (t_input, t_input, t_input))
    t_total = sum(t_res.values())

    # Keras
    k_model = keras.layers.MultiHeadAttention(
        num_heads=NumHeads, key_dim=EmbedDim // NumHeads
    )
    k_model.build(query_shape=(B, S, EmbedDim), value_shape=(B, S, EmbedDim))
    k_input = create_dummy_data_keras((S, EmbedDim), batch_size=B)

    # Keras Benchmark
    k_res = keras_bench.flop_count(k_model, ([k_input, k_input],))
    k_total = sum(k_res.values())

    # Compare TOTALS, not keys, because PyTorch splits MHA into atomic ops
    print(f"\nMHA Total FLOPs -> Torch: {t_total} | Keras: {k_total}")
    assert (
        abs(t_total - k_total) <= 1e-3
    ), f"MHA Total FLOP mismatch. T: {t_total}, K: {k_total}"


def test_group_norm_flops():
    # Common in Object Detection heads
    B, C, H, W = 2, 32, 16, 16
    Groups = 4

    # PyTorch
    t_model = nn.GroupNorm(num_groups=Groups, num_channels=C, affine=True)
    t_input = create_dummy_data_torch((C, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    # Keras
    k_model = keras.layers.GroupNormalization(
        groups=Groups, axis=-1, scale=True, center=True
    )
    k_model.build((B, H, W, C))
    k_input = create_dummy_data_keras((H, W, C), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # Mapping: Torch 'group_norm' -> Keras 'norm'
    # Note: Check if torch_bench outputs 'group_norm' or 'norm'
    t_key = "group_norm" if "group_norm" in t_res else "norm"
    assert_flops_near_equal(t_res, k_res, key_mapping={t_key: "norm"})


def test_strided_conv2d_flops():
    # Critical: Ensures output shape calculation logic is identical
    B, C_in, C_out = 1, 3, 16
    H, W = 64, 64
    K = 3
    Stride = 2

    # PyTorch (Padding=1 to match "same" when stride=1, but careful with Stride=2)
    # To match Keras "same" with Stride=2 exactly is tricky.
    # Easier to test "valid" padding to ensure math is pure.
    t_model = nn.Conv2d(C_in, C_out, K, stride=Stride, padding=0)
    t_input = create_dummy_data_torch((C_in, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    k_model = keras.layers.Conv2D(
        C_out, K, strides=Stride, padding="valid", data_format="channels_last"
    )
    k_model.build((B, H, W, C_in))
    k_input = create_dummy_data_keras((H, W, C_in), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    assert_flops_near_equal(t_res, k_res, key_mapping={"conv": "conv"})


def test_pooling_flops():
    B, C, H, W = 1, 64, 32, 32

    # PyTorch
    t_model = nn.MaxPool2d(kernel_size=2, stride=2)
    t_input = create_dummy_data_torch((C, H, W), batch_size=B)
    t_res = torch_bench.flop_count(t_model, (t_input,))

    # Keras
    k_model = keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )
    k_model.build((B, H, W, C))
    k_input = create_dummy_data_keras((H, W, C), batch_size=B)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # PyTorch benchmark usually classifies this as 'elementwise' or 'pooling'
    # benchmark.py classifies it as 'pooling'.
    # Check what key Torch returns (often 'max_pool2d' or 'elementwise' depending on config)
    t_key = list(t_res.keys())[0]
    assert_flops_near_equal(t_res, k_res, key_mapping={t_key: "pooling"})


def test_nested_model_traversal():
    """
    Ensures that benchmark.py correctly recurses into nested Subclassed models/layers.
    """

    # Define a simple nested Keras block
    class Block(keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(10)
            self.dense2 = keras.layers.Dense(10)

        def call(self, x):
            return self.dense2(self.dense1(x))

    # Define a top-level model using that block
    inputs = keras.Input(shape=(10,))
    x = Block()(inputs)
    x = keras.layers.Dense(5)(x)
    k_model = keras.Model(inputs, x)

    # Manually calculate expected FLOPs
    # Input (1, 10)
    # Dense1: 1 * 10 * 10 = 100
    # Dense2: 1 * 10 * 10 = 100
    # Dense3: 1 * 10 * 5  = 50
    # Total = 250

    k_input = create_dummy_data_keras((10,), batch_size=1)
    k_res = keras_bench.flop_count(k_model, (k_input,))

    # CORRECTION: Benchmark returns GFLOPs (1e9), not raw FLOPs
    expected_gflops = 250.0 / 1e9

    assert (
        abs(k_res["addmm"] - expected_gflops) < 1e-10
    ), f"Recursion failed. Expected {expected_gflops}, got {k_res.get('addmm')}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
