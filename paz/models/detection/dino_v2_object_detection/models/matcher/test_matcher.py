import pytest
import torch
import numpy as np
import os
import sys
import keras
from keras import ops


# Add project root to sys.path for 'examples' import
# Assumes we are running from project root or inside the file structure
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add project root to sys.path for 'examples' import
# Assumes we are running from project root or inside the file structure
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is .../paz/models/detection/dino_v2_object_detection/models/matcher
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add path to original implementation for 'rfdetr' import
rf_detr_path = os.path.join(
    project_root, "examples/rf-detr_original_pytorch_implementation"
)
if rf_detr_path not in sys.path:
    sys.path.append(rf_detr_path)

from rfdetr.models.matcher import HungarianMatcher as TorchMatcher


# Keras Implementation
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import (
    HungarianMatcher as KerasMatcher,
)
from paz.models.detection.dino_v2_object_detection.models.matcher.matcher_porting_utils import (
    convert_to_keras,
    assert_matcher_parity,
)


@pytest.fixture
def random_inputs():
    batch_size = 2
    num_queries = 10
    num_classes = 4

    # Random logits
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
    """
    Helper to run parity check with given matcher config and input config.
    """
    # Fix seed for deterministic tie-breaking in the cost matrix
    torch.manual_seed(42)
    np.random.seed(42)

    # Defaults
    cost_class = config.get("cost_class", 1)
    cost_bbox = config.get("cost_bbox", 1)
    cost_giou = config.get("cost_giou", 1)
    focal_alpha = config.get("focal_alpha", 0.25)

    group_detr = inputs_config.get("group_detr", 1)

    # Generate Inputs
    batch_size = inputs_config.get("batch_size", 2)
    num_queries = inputs_config.get("num_queries", 20)
    num_classes = inputs_config.get("num_classes", 4)
    empty_targets = inputs_config.get("empty_targets", False)

    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)

    targets = []
    for i in range(batch_size):
        if empty_targets and i == 1:  # Make second element empty
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

    # PyTorch Execution
    torch_matcher = TorchMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        focal_alpha=focal_alpha,
    )

    indices_torch = torch_matcher(outputs_torch, targets, group_detr=group_detr)

    # Keras Execution
    keras_matcher = KerasMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
        focal_alpha=focal_alpha,
    )

    # Convert inputs
    outputs_keras_np, targets_keras_np = convert_to_keras(outputs_torch, targets)

    indices_keras = keras_matcher(
        outputs_keras_np, targets_keras_np, group_detr=group_detr
    )

    # Verify
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
    run_parity_check(config)


def test_matcher_group_detr():
    # group_detr = 2
    # num_queries must be divisible
    run_parity_check(
        config={"cost_class": 1, "cost_bbox": 1, "cost_giou": 1},
        inputs_config={"group_detr": 2, "num_queries": 20},
    )


def test_matcher_empty_targets():
    # Verify handling of images with NO ground truth
    run_parity_check(
        config={"cost_class": 1, "cost_bbox": 1, "cost_giou": 1},
        inputs_config={"empty_targets": True, "batch_size": 3},
    )


def test_matcher_with_masks():
    batch_size = 1
    num_queries = 5
    num_classes = 2
    H, W = 20, 20

    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)
    pred_masks = torch.randn(batch_size, num_queries, H, W)  # mask logits

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

    # We execute checks just to ensure no crash and valid output structure
    indices_torch = torch_matcher(outputs_torch, targets)

    # Keras
    # Keras
    outputs_keras_np, targets_keras_np = convert_to_keras(outputs_torch, targets)

    keras_matcher = KerasMatcher(
        cost_class=1, cost_bbox=5, cost_giou=2, cost_mask_ce=1, cost_mask_dice=1
    )

    indices_keras = keras_matcher(outputs_keras_np, targets_keras_np)

    assert len(indices_keras) == batch_size
    for r, c in indices_keras:
        r_np = keras.ops.convert_to_numpy(r)
        c_np = keras.ops.convert_to_numpy(c)
        assert len(r_np) == len(c_np)
        assert len(np.unique(r_np)) == len(r_np)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
