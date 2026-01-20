import pytest
import numpy as np
import torch
import keras
from keras import ops
import os
import sys

# -------------------------------------------------------------------------
# 0. Environment Setup
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
sys.path.append(project_root)
print(f"Project Root: {project_root}")


# Import the modules
from examples.dino_object_detection.models.utils import misc as keras_misc
from examples.dino_object_detection.models.utils import (
    torch_misc_for_testing as torch_misc,
)

# --- Helpers ---


def to_numpy(x):
    """Converts PyTorch tensors, Keras tensors, or lists to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def assert_close(keras_res, torch_res, rtol=1e-5, atol=1e-5):
    """Asserts that Keras and Torch results are numerically close."""
    k_np = to_numpy(keras_res)
    t_np = to_numpy(torch_res)
    np.testing.assert_allclose(k_np, t_np, rtol=rtol, atol=atol)


def get_random_inputs(shape, seed=42):
    """Generates identical random data for both frameworks."""
    np.random.seed(seed)
    data = np.random.randn(*shape).astype(np.float32)
    k_data = ops.convert_to_tensor(data)
    t_data = torch.tensor(data)
    return k_data, t_data


# --- Tests ---


def test_inverse_sigmoid():
    np.random.seed(1)
    data = np.random.rand(5, 5).astype(np.float32)

    k_in = ops.convert_to_tensor(data)
    t_in = torch.tensor(data)

    k_out = keras_misc.inverse_sigmoid(k_in)
    t_out = torch_misc.inverse_sigmoid(t_in)

    assert_close(k_out, t_out)


def test_smoothed_value_logic():
    k_smooth = keras_misc.SmoothedValue(window_size=5)
    t_smooth = torch_misc.SmoothedValue(window_size=5)

    values = [0.5, 1.2, 0.8, 3.0, 0.1, 0.9]
    for v in values:
        k_smooth.update(v)
        t_smooth.update(v)

    assert np.isclose(k_smooth.median, t_smooth.median)
    assert np.isclose(k_smooth.avg, t_smooth.avg)
    assert np.isclose(k_smooth.global_avg, t_smooth.global_avg)
    assert np.isclose(k_smooth.max, t_smooth.max)
    assert np.isclose(k_smooth.value, t_smooth.value)


def test_accuracy_top1():
    k_logits, t_logits = get_random_inputs((4, 10))
    targets_np = np.array([0, 2, 9, 1], dtype=np.int64)

    k_target = ops.convert_to_tensor(targets_np)
    t_target = torch.tensor(targets_np)

    k_acc = keras_misc.accuracy(k_logits, k_target, topk=(1,))
    t_acc = torch_misc.accuracy(t_logits, t_target, topk=(1,))

    assert_close(k_acc[0], t_acc[0])


def test_accuracy_topk_multiple():
    # This previously failed in PyTorch code.
    k_logits, t_logits = get_random_inputs((10, 20))
    targets_np = np.random.randint(0, 20, size=(10,))

    k_target = ops.convert_to_tensor(targets_np)
    t_target = torch.tensor(targets_np)

    topk = (1, 5)
    k_res = keras_misc.accuracy(k_logits, k_target, topk=topk)
    t_res = torch_misc.accuracy(t_logits, t_target, topk=topk)

    for k_r, t_r in zip(k_res, t_res):
        assert_close(k_r, t_r)


def test_interpolate_nearest():
    # This previously failed due to mismatch.
    # Shape: (B, C, H, W) -> (1, 3, 20, 20)
    k_img, t_img = get_random_inputs((1, 3, 20, 20))

    size = (30, 30)  # Upsample

    k_out = keras_misc.interpolate(k_img, size=size, mode="nearest")
    t_out = torch_misc.interpolate(t_img, size=size, mode="nearest")

    assert_close(k_out, t_out)


def test_interpolate_scaling():
    k_img, t_img = get_random_inputs((1, 1, 10, 10))

    scale = 2.0
    k_out = keras_misc.interpolate(k_img, scale_factor=scale, mode="nearest")
    t_out = torch_misc.interpolate(t_img, scale_factor=scale, mode="nearest")

    assert k_out.shape == (1, 1, 20, 20)
    assert_close(k_out, t_out)


def test_nested_tensor_packing():
    shapes = [(3, 10, 10), (3, 15, 20), (3, 12, 12)]
    np_images = [np.random.randn(*s).astype(np.float32) for s in shapes]

    k_list = [ops.convert_to_tensor(img) for img in np_images]
    t_list = [torch.tensor(img) for img in np_images]

    k_nested = keras_misc.nested_tensor_from_tensor_list(k_list)
    t_nested = torch_misc.nested_tensor_from_tensor_list(t_list)

    k_tensors, k_mask = k_nested.decompose()
    t_tensors, t_mask = t_nested.decompose()

    expected_shape = (3, 3, 15, 20)

    assert tuple(k_tensors.shape) == expected_shape
    assert tuple(t_tensors.shape) == expected_shape

    assert_close(k_tensors, t_tensors)
    assert_close(k_mask, t_mask)


def test_nested_tensor_mask_logic():
    img_data = np.ones((1, 2, 2), dtype=np.float32)
    k_list = [ops.convert_to_tensor(img_data)]
    large_img = np.ones((1, 4, 4), dtype=np.float32)
    k_list.append(ops.convert_to_tensor(large_img))

    k_nested = keras_misc.nested_tensor_from_tensor_list(k_list)
    _, k_mask = k_nested.decompose()

    k_mask_np = to_numpy(k_mask)

    # Padding check: (0,0) valid, (3,3) padded
    assert k_mask_np[0, 0, 0] == 0
    assert k_mask_np[0, 3, 3] == 1
    assert np.all(k_mask_np[1] == 0)


def test_metric_logger_basic():
    k_logger = keras_misc.MetricLogger()
    t_logger = torch_misc.MetricLogger()

    metrics = {"loss": 0.5, "acc": 90.0}

    k_logger.update(**metrics)
    t_logger.update(**metrics)

    assert k_logger.loss.count == t_logger.loss.count
    assert np.isclose(k_logger.loss.global_avg, t_logger.loss.global_avg)
    assert "loss" in str(k_logger)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
