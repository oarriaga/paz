import pytest
import numpy as np
import torch
import keras
import os
import sys

# -------------------------------------------------------------------------
# 0. Environment Setup
# -------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
sys.path.append(project_root)
print(f"Project Root: {project_root}")

from examples.dino_object_detection.models.matcher import (
    HungarianMatcher as KerasMatcher,
)
from examples.dino_object_detection.models.torch_matcher_for_testing import (
    HungarianMatcher as TorchMatcher,
)


def get_test_data(bs=2, num_queries=10, num_classes=4, num_targets=3, with_masks=False):
    """Generates random data consistent for both frameworks."""
    np.random.seed(42)

    # Outputs
    out_logits = np.random.randn(bs, num_queries, num_classes).astype(np.float32)
    out_boxes = np.random.rand(bs, num_queries, 4).astype(np.float32)  # cxcywh
    out_masks = None
    if with_masks:
        out_masks = np.random.randn(bs, num_queries, 32, 32).astype(np.float32)

    # Targets
    targets = []
    for _ in range(bs):
        n_tgt = np.random.randint(1, num_targets + 1)
        t = {
            "labels": np.random.randint(0, num_classes, (n_tgt,)).astype(np.int64),
            "boxes": np.random.rand(n_tgt, 4).astype(np.float32),
        }
        if with_masks:
            t["masks"] = np.random.randint(0, 2, (n_tgt, 32, 32)).astype(np.float32)
        targets.append(t)

    return out_logits, out_boxes, out_masks, targets


def to_torch(logits, boxes, masks, targets):
    out = {
        "pred_logits": torch.from_numpy(logits),
        "pred_boxes": torch.from_numpy(boxes),
    }
    if masks is not None:
        out["pred_masks"] = torch.from_numpy(masks)

    tgt_list = []
    for t in targets:
        d = {k: torch.from_numpy(v) for k, v in t.items()}
        tgt_list.append(d)
    return out, tgt_list


def to_keras(logits, boxes, masks, targets):
    out = {
        "pred_logits": keras.ops.convert_to_tensor(logits),
        "pred_boxes": keras.ops.convert_to_tensor(boxes),
    }
    if masks is not None:
        out["pred_masks"] = keras.ops.convert_to_tensor(masks)

    tgt_list = []
    for t in targets:
        d = {k: keras.ops.convert_to_tensor(v) for k, v in t.items()}
        tgt_list.append(d)
    return out, tgt_list


def check_equivalence(k_res, t_res):
    """Asserts that index tuples from both matchers are identical."""
    assert len(k_res) == len(t_res)
    for i in range(len(k_res)):
        # Convert Keras results to numpy
        k_src = keras.ops.convert_to_numpy(k_res[i][0])
        k_tgt = keras.ops.convert_to_numpy(k_res[i][1])

        # Convert Torch results to numpy
        t_src = t_res[i][0].numpy()
        t_tgt = t_res[i][1].numpy()

        np.testing.assert_array_equal(
            k_src, t_src, err_msg=f"Batch {i} source indices mismatch"
        )
        np.testing.assert_array_equal(
            k_tgt, t_tgt, err_msg=f"Batch {i} target indices mismatch"
        )


# --- Tests ---


def test_basic_matching():
    """Verify exact match for standard bbox + label tasks."""
    logits, boxes, _, targets = get_test_data(bs=4, num_queries=15)

    # PyTorch Reference
    t_out, t_tgt = to_torch(logits, boxes, None, targets)
    t_matcher = TorchMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    t_res = t_matcher(t_out, t_tgt)

    # Keras Implementation
    k_out, k_tgt = to_keras(logits, boxes, None, targets)
    k_matcher = KerasMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    k_res = k_matcher(k_out, k_tgt)

    check_equivalence(k_res, t_res)


def test_group_detr_matching():
    """Verify equivalence when using group_detr parameter."""
    group_detr = 2
    # Ensure num_queries is divisible by group_detr
    logits, boxes, _, targets = get_test_data(bs=2, num_queries=20)

    # PyTorch
    t_out, t_tgt = to_torch(logits, boxes, None, targets)
    t_matcher = TorchMatcher()
    t_res = t_matcher(t_out, t_tgt, group_detr=group_detr)

    # Keras
    k_out, k_tgt = to_keras(logits, boxes, None, targets)
    k_matcher = KerasMatcher()
    k_res = k_matcher(k_out, k_tgt, group_detr=group_detr)

    check_equivalence(k_res, t_res)


def test_empty_targets():
    """Verify robustness with empty targets."""
    logits, boxes, _, _ = get_test_data(bs=2)
    # Manually clear targets for the first batch item
    targets = [
        {
            "labels": np.array([], dtype=np.int64),
            "boxes": np.zeros((0, 4), dtype=np.float32),
        },
        {
            "labels": np.array([1], dtype=np.int64),
            "boxes": np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
        },
    ]

    # PyTorch
    t_out, t_tgt = to_torch(logits, boxes, None, targets)
    t_matcher = TorchMatcher()
    t_res = t_matcher(t_out, t_tgt)

    # Keras
    k_out, k_tgt = to_keras(logits, boxes, None, targets)
    k_matcher = KerasMatcher()
    k_res = k_matcher(k_out, k_tgt)

    check_equivalence(k_res, t_res)


def test_masks_integration_runs():
    """
    Verify Keras mask matching runs without error and produces valid shapes.
    Note: Cannot strictly compare indices with Torch due to internal RNG differences
    in point sampling, but this proves the logic pipeline is valid.
    """
    logits, boxes, masks, targets = get_test_data(bs=2, with_masks=True)

    k_out, k_tgt = to_keras(logits, boxes, masks, targets)
    k_matcher = KerasMatcher(cost_mask_ce=1.0, cost_mask_dice=1.0)
    k_res = k_matcher(k_out, k_tgt)

    # Check that we got matches for each target
    assert len(k_res) == 2
    for i in range(2):
        n_tgt = len(targets[i]["boxes"])
        assert k_res[i][0].shape[0] == n_tgt


if __name__ == "__main__":
    pytest.main(["-v", __file__])
