import pytest
import torch
import numpy as np
import os
import sys
import keras
from keras import ops


# Resolve project root and reference implementation paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

rf_detr_path = os.path.join(
    project_root, "examples/rf-detr_original_pytorch_implementation"
)
if rf_detr_path not in sys.path:
    sys.path.append(rf_detr_path)

from rfdetr.models.matcher import HungarianMatcher as TorchMatcher

from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import (
    HungarianMatcher as KerasMatcher,
)
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher_porting_utils import (
    convert_to_keras,
    assert_matcher_parity,
)


@pytest.fixture
def random_inputs():
    """Generates random prediction and target tensors for matcher testing.

    Returns:
        dict: Contains pred_logits, pred_boxes, and a list of target dicts.
    """
    batch_size = 2
    num_queries = 10
    num_classes = 4

    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)

    targets = []
    for _ in range(batch_size):
        num_targets = np.random.randint(1, 5)
        targets.append(
            {
                "labels": torch.randint(0, num_classes, (num_targets,)),
                "boxes": torch.rand(num_targets, 4),
            }
        )

    return {"pred_logits": pred_logits, "pred_boxes": pred_boxes, "targets": targets}


def run_parity_check(config, inputs_config={}):
    """Runs a parity check between reference and tested matcher.

    Creates matcher instances with the given configuration, generates
    random inputs, runs both matchers, and asserts output equivalence.

    Args:
        config (dict): Matcher cost weights (cost_class, cost_bbox,
            cost_giou, focal_alpha).
        inputs_config (dict): Input generation parameters (batch_size,
            num_queries, num_classes, group_detr, empty_targets).
    """
    # Fix seed for deterministic tie-breaking in the cost matrix
    torch.manual_seed(42)
    np.random.seed(42)

    cost_class = config.get("cost_class", 1)
    cost_bbox = config.get("cost_bbox", 1)
    cost_giou = config.get("cost_giou", 1)
    focal_alpha = config.get("focal_alpha", 0.25)

    group_detr = inputs_config.get("group_detr", 1)

    batch_size = inputs_config.get("batch_size", 2)
    num_queries = inputs_config.get("num_queries", 20)
    num_classes = inputs_config.get("num_classes", 4)
    empty_targets = inputs_config.get("empty_targets", False)

    # Generate random prediction tensors
    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)

    # Generate random per-image targets with variable counts
    targets = []
    for i in range(batch_size):
        if empty_targets and i == 1:
            num_targets = 0
        else:
            num_targets = np.random.randint(1, 5)

        targets.append(
            {
                "labels": torch.randint(0, num_classes, (num_targets,)),
                "boxes": torch.rand(num_targets, 4),
            }
        )

    outputs_torch = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    # Run reference matcher
    torch_matcher = TorchMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        focal_alpha=focal_alpha,
    )

    indices_torch = torch_matcher(outputs_torch, targets, group_detr=group_detr)

    # Run tested matcher
    keras_matcher = KerasMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        focal_alpha=focal_alpha,
    )

    # Convert inputs to numpy format for the tested matcher
    outputs_keras_np, targets_keras_np = convert_to_keras(outputs_torch, targets)

    indices_keras = keras_matcher(
        outputs_keras_np, targets_keras_np, group_detr=group_detr
    )

    # Verify index-level equivalence
    assert_matcher_parity(indices_torch, indices_keras)


@pytest.mark.parametrize(
    "config",
    [
        {"cost_class": 1, "cost_bbox": 5, "cost_giou": 2},  # Default-ish
        {"cost_class": 0, "cost_bbox": 1, "cost_giou": 1},  # Box only
        {"cost_class": 1, "cost_bbox": 0, "cost_giou": 0},  # Class only
        {
            "cost_class": 1,
            "cost_bbox": 1,
            "cost_giou": 1,
            "focal_alpha": 0.9,
        },  # High alpha
        {"cost_class": 2.5, "cost_bbox": 0.5, "cost_giou": 10},  # Varied weights
    ],
)
def test_matcher_weight_configurations(config):
    """Tests matcher parity across different cost weight configurations."""
    run_parity_check(config)


def test_matcher_group_detr():
    """Tests matcher parity with group DETR (num_queries split into groups)."""
    run_parity_check(
        config={"cost_class": 1, "cost_bbox": 1, "cost_giou": 1},
        inputs_config={"group_detr": 2, "num_queries": 20},
    )


def test_matcher_empty_targets():
    """Tests matcher handling when some images have no ground-truth targets."""
    run_parity_check(
        config={"cost_class": 1, "cost_bbox": 1, "cost_giou": 1},
        inputs_config={"empty_targets": True, "batch_size": 3},
    )


def test_matcher_with_masks():
    """Tests matcher with mask predictions and targets.

    Verifies that the matcher runs without error, returns valid structure,
    and produces unique row assignments when masks are included in the
    cost computation.
    """
    batch_size = 1
    num_queries = 5
    num_classes = 2
    H, W = 20, 20

    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)
    pred_masks = torch.randn(batch_size, num_queries, H, W)

    targets = []
    for _ in range(batch_size):
        num_targets = 2
        targets.append(
            {
                "labels": torch.randint(0, num_classes, (num_targets,)),
                "boxes": torch.rand(num_targets, 4),
                "masks": torch.randint(0, 2, (num_targets, H, W)).float(),
            }
        )

    outputs_torch = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
    }

    torch_matcher = TorchMatcher(
        cost_class=1, cost_bbox=5, cost_giou=2, cost_mask_ce=1, cost_mask_dice=1
    )

    # Run reference matcher to ensure no crash
    indices_torch = torch_matcher(outputs_torch, targets)

    # Run tested matcher
    outputs_keras_np, targets_keras_np = convert_to_keras(outputs_torch, targets)

    keras_matcher = KerasMatcher(
        cost_class=1, cost_bbox=5, cost_giou=2, cost_mask_ce=1, cost_mask_dice=1
    )

    indices_keras = keras_matcher(outputs_keras_np, targets_keras_np)

    # Verify output structure: one result per batch element with
    # unique row assignments
    assert len(indices_keras) == batch_size
    for r, c in indices_keras:
        r_np = keras.ops.convert_to_numpy(r)
        c_np = keras.ops.convert_to_numpy(c)
        assert len(r_np) == len(c_np)
        assert len(np.unique(r_np)) == len(r_np)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
