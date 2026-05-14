import pytest
import torch
import numpy as np
import keras
import sys
import os

# Resolve project root and reference implementation paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../examples/rf-detr_original_pytorch_implementation")))

from rfdetr import (
    RFDETRSmall,
    RFDETRMedium,
    RFDETRNano,
    RFDETRLarge,
    RFDETRBase,
    RFDETRSegNano,
    RFDETRSegSmall,
)
from rfdetr.models.matcher import build_matcher as build_torch_matcher
try:
    from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
except ImportError:
    import rfdetr.util.box_ops as box_ops
    box_cxcywh_to_xyxy = box_ops.box_cxcywh_to_xyxy
    generalized_box_iou = box_ops.generalized_box_iou

from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher as KerasHungarianMatcher

from paz.models.detection.dino_v2_object_detection.models.matcher.matcher_porting_utils import (
    to_numpy, convert_to_keras, extract_matcher_config, 
    build_keras_matcher_from_config, assert_matcher_parity
)

# Model variants to test against
MODELS_TO_TEST = [
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRBase,
    RFDETRLarge,
    RFDETRSegNano, 
    RFDETRSegSmall,
]

def compute_pytorch_cost_matrix(outputs, targets, matcher):
    """Computes the reference cost matrix for verification.

    Replicates the cost matrix calculation logic using the reference
    implementation's box ops for direct numerical comparison with the
    tested matcher's cost matrix.

    Args:
        outputs (dict): Model predictions with "pred_logits" and
            "pred_boxes" tensors.
        targets (list[dict]): Per-image target dicts with "labels" and
            "boxes" tensors.
        matcher: Reference matcher instance providing cost weights and
            focal_alpha.

    Returns:
        Tensor: Cost matrix of shape (B, Q, total_targets).
    """
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten predictions for pairwise computation
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Concatenate all targets across batch
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Generalized IoU cost (negated for minimization)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Focal loss classification cost
        alpha = matcher.focal_alpha
        gamma = 2.0
        
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-torch.nn.functional.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-torch.nn.functional.logsigmoid(flat_pred_logits))
        
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # L1 bounding box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Weighted combination of all cost terms
        C = matcher.cost_bbox * cost_bbox + matcher.cost_class * cost_class + matcher.cost_giou * cost_giou
        
        C = C.view(bs, num_queries, -1).cpu()
        return C

@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
def test_matcher_config_parity(model_class):
    """Tests matcher output parity using real model configurations.

    Initializes a model to extract its configuration, builds both
    reference and tested matchers, and verifies identical outputs.
    """
    _run_matcher_check(model_class, check_cost_matrix=True)

@pytest.mark.parametrize("overrides", [
    {"set_cost_class": 5.0, "set_cost_bbox": 0.0, "set_cost_giou": 0.0}, # Class only
    {"set_cost_class": 0.0, "set_cost_bbox": 5.0, "set_cost_giou": 0.0},
    {"set_cost_class": 0.0, "set_cost_bbox": 0.0, "set_cost_giou": 5.0},
    {"group_detr": 3},
])
def test_matcher_custom_configs(overrides):
    """Tests matcher parity with custom configuration overrides."""
    _run_matcher_check(RFDETRSmall, config_overrides=overrides, check_cost_matrix=True)

def test_matcher_empty_targets():
    """Tests matcher robustness when one image has no targets."""
    _run_matcher_check(RFDETRSmall, empty_targets=True, check_cost_matrix=False)

def _run_matcher_check(model_class, config_overrides=None, empty_targets=False, check_cost_matrix=False):
    """Runs a full matcher parity check for a given model class.

    Instantiates the model to extract its configuration, builds both
    reference and tested matchers, generates dummy data matching the
    model's query/class dimensions, and compares outputs.

    Args:
        model_class: Model class to instantiate for config extraction.
        config_overrides (dict): Optional parameter overrides to apply.
        empty_targets (bool): If True, first image has zero targets.
        check_cost_matrix (bool): If True, also verifies cost matrix
            numerical equivalence (skipped for segmentation models due
            to random mask point sampling).
    """
    if config_overrides is None:
        config_overrides = {}

    print(f"\nTesting Matcher Parity for model: {model_class.__name__}")
    print(f"  Overrides: {config_overrides}")
    
    # Instantiate model to extract its configuration
    try:
        rfdetr_wrapper = model_class(pretrain_weights=None)
    except Exception as e:
        print(f"Skipping instantiation with pretrain_weights=None, trying default: {e}")
        rfdetr_wrapper = model_class()

    args = rfdetr_wrapper.model.args
    
    # Apply configuration overrides
    for k, v in config_overrides.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            setattr(args, k, v)

    # Inject default mask loss coefficients for segmentation models
    if getattr(args, 'segmentation_head', False):
        if not hasattr(args, 'mask_ce_loss_coef'):
            args.mask_ce_loss_coef = 5.0 
        if not hasattr(args, 'mask_dice_loss_coef'):
            args.mask_dice_loss_coef = 5.0
        if not hasattr(args, 'mask_point_sample_ratio'):
            args.mask_point_sample_ratio = 16

    # Build reference matcher from model args
    torch_matcher = build_torch_matcher(args)
    
    # Build tested matcher using extracted config
    config = extract_matcher_config(args)
    keras_matcher = build_keras_matcher_from_config(config)
    
    print(f"  Config detected: Class={keras_matcher.cost_class}, Box={keras_matcher.cost_bbox}, "
          f"GIoU={keras_matcher.cost_giou}, Alpha={keras_matcher.focal_alpha}")

    # Generate dummy data matching the model's dimensions
    batch_size = 2
    num_queries = args.num_queries * args.group_detr
    num_classes = args.num_classes 
    
    # Prediction tensors: logits are unconstrained, boxes are in [0, 1]
    outputs_torch = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)), 
    }
    
    # Per-image targets with variable number of ground-truth objects
    targets_torch = []
    for i in range(batch_size):
        if empty_targets and i == 0:
            n_boxes = 0
        else:
            n_boxes = np.random.randint(1, 10)
            
        targets_torch.append({
            "labels": torch.randint(0, num_classes, (n_boxes,)).long(),
            "boxes": torch.rand(n_boxes, 4), 
        })

    # Add mask data for segmentation models
    if getattr(args, 'segmentation_head', False):
        mask_h, mask_w = 32, 32
        outputs_torch["pred_masks"] = torch.randn(batch_size, num_queries, mask_h, mask_w)
        
        for i in range(batch_size):
            n_boxes = targets_torch[i]["boxes"].shape[0]
            if n_boxes > 0:
                targets_torch[i]["masks"] = torch.randint(0, 2, (n_boxes, mask_h, mask_w)).float()
            else:
                 targets_torch[i]["masks"] = torch.zeros((0, mask_h, mask_w)).float()

    # Run reference matcher
    with torch.no_grad():
        indices_torch = torch_matcher(outputs_torch, targets_torch, group_detr=args.group_detr)
        
        # Optionally verify cost matrix numerical equivalence
        # (skipped for segmentation models due to random mask point sampling)
        if check_cost_matrix and not getattr(args, 'segmentation_head', False): 
            batch_C_torch = compute_pytorch_cost_matrix(outputs_torch, targets_torch, torch_matcher)

    # Run tested matcher
    outputs_keras, targets_keras = convert_to_keras(outputs_torch, targets_torch)
    
    # Verify cost matrix equivalence when applicable
    if check_cost_matrix and not getattr(args, 'segmentation_head', False):
        batch_C_keras = keras_matcher.compute_cost_matrix(outputs_keras, targets_keras)
        # Reshape from flat (B*Q, total_targets) to (B, Q, total_targets)
        # to match the reference cost matrix layout
        batch_C_keras_np = to_numpy(batch_C_keras).reshape(batch_size, num_queries, -1)
        batch_C_torch_np = to_numpy(batch_C_torch)
        
        # Assert numerical closeness within tolerance
        print("  Verifying Cost Matrix values...")
        np.testing.assert_allclose(batch_C_keras_np, batch_C_torch_np, rtol=0, atol=1e-4, err_msg="Cost Matrix mismatch")
        print("  Cost Matrix Parity Confirmed (1e-4 tolerance).")

    indices_keras = keras_matcher(outputs_keras, targets_keras, group_detr=args.group_detr)

    # Compare matched indices
    # For segmentation models, skip exact check due to random mask sampling
    check_exact = not getattr(args, 'segmentation_head', False)
    
    try:
        assert_matcher_parity(indices_torch, indices_keras, check_exact=check_exact)
        if not check_exact:
             print(f"  Segmentation model {model_class.__name__}: Skipped exact check due to random mask sampling. Structure valid.")
        else:
             print(f"  Model {model_class.__name__}: Exact parity confirmed for all batches.")
    except AssertionError as e:
        # If exact parity fails, check if the total assignment costs are
        # equivalent (differences may arise from tie-breaking in the
        # linear assignment solver when multiple optimal solutions exist)
        if check_cost_matrix and 'batch_C_torch' in locals():
            print(f"  Exact parity failed ({e}). Checking assignment cost parity (tolerance 1e-4)...")
            batch_C_torch_np = to_numpy(batch_C_torch)
            
            for i in range(len(indices_torch)):
                r_t, c_t = to_numpy(indices_torch[i][0]), to_numpy(indices_torch[i][1])
                r_k, c_k = to_numpy(indices_keras[i][0]), to_numpy(indices_keras[i][1])
                
                # Evaluate both assignments on the reference cost matrix
                # to determine if the total cost is equivalent
                cost_torch = batch_C_torch_np[i][r_t, c_t].sum()
                cost_keras = batch_C_torch_np[i][r_k, c_k].sum()
                
                diff = abs(cost_torch - cost_keras)
                if diff > 1e-4:
                    raise AssertionError(f"Cost mismatch at zero-tolerance! Batch {i}: Torch={cost_torch}, Keras={cost_keras}, Diff={diff}")
            
            print("  Assignment Cost Parity Confirmed (1e-4). Mismatch purely due to tie-breaking.")
        else:
            raise

if __name__ == "__main__":
    pytest.main([__file__])
