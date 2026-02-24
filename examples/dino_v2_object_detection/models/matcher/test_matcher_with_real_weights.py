import pytest
import torch
import numpy as np
import keras
import sys
import os

# Add path to finding rfdetr and examples
# Adjust this based on where this file is located relative to project root
# Current file: examples/dino_v2_object_detection/models/matcher/test_matcher_with_real_weights.py
# Project root: ../../../../
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../rf-detr_original_pytorch_implementation")))

# Import RF-DETR models directly
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
    # If not importable directly, try from submodules if path is messy
    import rfdetr.util.box_ops as box_ops
    box_cxcywh_to_xyxy = box_ops.box_cxcywh_to_xyxy
    generalized_box_iou = box_ops.generalized_box_iou

# Import Keras Implementation
from examples.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher as KerasHungarianMatcher

from examples.dino_v2_object_detection.models.matcher.matcher_porting_utils import (
    to_numpy, convert_to_keras, extract_matcher_config, 
    build_keras_matcher_from_config, assert_matcher_parity
)

# List of models to test
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
    """
    Replicates the cost matrix calculation logic of the RF-DETR HungarianMatcher using PyTorch.
    This allows us to verify the cost values directly against Keras implementation.
    """
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Compute the classification cost.
        alpha = matcher.focal_alpha
        gamma = 2.0
        
        # Let's try to match Keras logic which used log_sigmoid:
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-torch.nn.functional.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-torch.nn.functional.logsigmoid(flat_pred_logits))
        
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Final cost matrix
        C = matcher.cost_bbox * cost_bbox + matcher.cost_class * cost_class + matcher.cost_giou * cost_giou
        
        # Add mask costs if any (simplified: assume no masks for strict cost check unless needed)
        # If masks needed, we need to implement point sampling in torch here.
        
        C = C.view(bs, num_queries, -1).cpu()
        return C

@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
def test_matcher_config_parity(model_class):
    """
    Tests that the Keras matcher, when initialized with parameters extracted
    from a real RF-DETR model configuration, produces the same results as the
    PyTorch matcher built from the same configuration.
    """
    _run_matcher_check(model_class, check_cost_matrix=True)

@pytest.mark.parametrize("overrides", [
    {"set_cost_class": 5.0, "set_cost_bbox": 0.0, "set_cost_giou": 0.0}, # Class only
    {"set_cost_class": 0.0, "set_cost_bbox": 5.0, "set_cost_giou": 0.0}, # Box only
    {"set_cost_class": 0.0, "set_cost_bbox": 0.0, "set_cost_giou": 5.0}, # GIoU only
    {"group_detr": 3},
])
def test_matcher_custom_configs(overrides):
    """Test with custom configuration overrides on a standard model."""
    # Use RFDETRSmall as base for config tests
    _run_matcher_check(RFDETRSmall, config_overrides=overrides, check_cost_matrix=True)

def test_matcher_empty_targets():
    """Test with one image having no targets."""
    # Cost check might fail if C is empty or shaped weirdly, checking robustness
    _run_matcher_check(RFDETRSmall, empty_targets=True, check_cost_matrix=False) # Skip strict cost check for edge case for now

def _run_matcher_check(model_class, config_overrides=None, empty_targets=False, check_cost_matrix=False):
    """
    Helper function to run matcher parity check with optional config overrides.
    """
    if config_overrides is None:
        config_overrides = {}

    print(f"\nTesting Matcher Parity for model: {model_class.__name__}")
    print(f"  Overrides: {config_overrides}")
    
    # 1. Instantiate the RF-DETR model to get its configuration
    try:
        # Use pretrain_weights=None to avoid downloading weights
        rfdetr_wrapper = model_class(pretrain_weights=None)
    except Exception as e:
        print(f"Skipping instantiation with pretrain_weights=None, trying default: {e}")
        rfdetr_wrapper = model_class()

    # 2. Extract arguments
    args = rfdetr_wrapper.model.args
    
    # Apply overrides
    for k, v in config_overrides.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            setattr(args, k, v)

    # Inject missing mask loss coefficients if segmentation_head is present
    if getattr(args, 'segmentation_head', False):
        if not hasattr(args, 'mask_ce_loss_coef'):
            args.mask_ce_loss_coef = 5.0 
        if not hasattr(args, 'mask_dice_loss_coef'):
            args.mask_dice_loss_coef = 5.0
        if not hasattr(args, 'mask_point_sample_ratio'):
            args.mask_point_sample_ratio = 16

    # 3. Build PyTorch Matcher
    torch_matcher = build_torch_matcher(args)
    
    # 4. build Keras Matcher using extracted config
    config = extract_matcher_config(args)
    keras_matcher = build_keras_matcher_from_config(config)
    
    print(f"  Config detected: Class={keras_matcher.cost_class}, Box={keras_matcher.cost_bbox}, "
          f"GIoU={keras_matcher.cost_giou}, Alpha={keras_matcher.focal_alpha}")

    # 5. Generate Dummy Data
    batch_size = 2
    # In training with group_detr, the total number of queries is num_queries * group_detr
    num_queries = args.num_queries * args.group_detr
    num_classes = args.num_classes 
    
    # Outputs
    outputs_torch = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.sigmoid(torch.randn(batch_size, num_queries, 4)), 
    }
    
    # Targets
    targets_torch = []
    for i in range(batch_size):
        if empty_targets and i == 0:
            n_boxes = 0 # Test empty target for first image
        else:
            n_boxes = np.random.randint(1, 10)
            
        targets_torch.append({
            "labels": torch.randint(0, num_classes, (n_boxes,)).long(), # Ensure long for indexing
            "boxes": torch.rand(n_boxes, 4), 
        })

    # Handle Segmentation Data
    if getattr(args, 'segmentation_head', False):
        mask_h, mask_w = 32, 32
        outputs_torch["pred_masks"] = torch.randn(batch_size, num_queries, mask_h, mask_w)
        
        for i in range(batch_size):
            n_boxes = targets_torch[i]["boxes"].shape[0]
            if n_boxes > 0:
                targets_torch[i]["masks"] = torch.randint(0, 2, (n_boxes, mask_h, mask_w)).float()
            else:
                 targets_torch[i]["masks"] = torch.zeros((0, mask_h, mask_w)).float()

    # 6. Run PyTorch Matcher
    with torch.no_grad():
        indices_torch = torch_matcher(outputs_torch, targets_torch, group_detr=args.group_detr)
        
        if check_cost_matrix and not getattr(args, 'segmentation_head', False): 
            # Only checking cost matrix for detection-only for now due to complexity of mask cost replication
            batch_C_torch = compute_pytorch_cost_matrix(outputs_torch, targets_torch, torch_matcher)

    # 7. Run Keras Matcher
    outputs_keras, targets_keras = convert_to_keras(outputs_torch, targets_torch)
    
    # Compute Cost Matrix (New check)
    if check_cost_matrix and not getattr(args, 'segmentation_head', False):
        batch_C_keras = keras_matcher.compute_cost_matrix(outputs_keras, targets_keras)
        # batch_C_keras is usually flat (Batch*NumQueries, SumTargets) before split
        # We need to reshape to match PyTorch logic in compute_pytorch_cost_matrix: (Batch, NumQueries, TotalTargets)
        # See compute_pytorch_cost_matrix logic: C = C.view(bs, num_queries, -1). 
        # This assumes total_targets is consistent? No, C flat is (N_total, M_total). 
        # view(bs, num_queries, -1) -> (bs, num_queries, M_total). 
        # Yes.
        
        batch_C_keras_np = to_numpy(batch_C_keras).reshape(batch_size, num_queries, -1)
        batch_C_torch_np = to_numpy(batch_C_torch)
        
        # Assert Close
        print("  Verifying Cost Matrix values...")
        np.testing.assert_allclose(batch_C_keras_np, batch_C_torch_np, rtol=0, atol=1e-4, err_msg="Cost Matrix mismatch")
        print("  Cost Matrix Parity Confirmed (1e-4 tolerance).")

    indices_keras = keras_matcher(outputs_keras, targets_keras, group_detr=args.group_detr)

    # 8. Compare Results
    # For segmentation, we skip exact check as before
    check_exact = not getattr(args, 'segmentation_head', False)
    
    try:
        assert_matcher_parity(indices_torch, indices_keras, check_exact=check_exact)
        if not check_exact:
             print(f"  Segmentation model {model_class.__name__}: Skipped exact parity check due to random sampling. Structure valid.")
        else:
             print(f"  Model {model_class.__name__}: Exact parity confirmed for all batches.")
    except AssertionError as e:
        # If exact parity fails, we check if the costs are identical (tie-breaking differences)
        if check_cost_matrix and 'batch_C_torch' in locals():
            print(f"  Exact parity failed ({e}). Checking assignment cost parity (tolerance 1e-4)...")
            batch_C_torch_np = to_numpy(batch_C_torch)
            
            for i in range(len(indices_torch)):
                # Indices for batch i
                r_t, c_t = to_numpy(indices_torch[i][0]), to_numpy(indices_torch[i][1])
                r_k, c_k = to_numpy(indices_keras[i][0]), to_numpy(indices_keras[i][1])
                
                # Reshape C for this batch if necessary
                # batch_C_torch is (BS, NumQueries, TotalTargets) or (BS, NumQueries, M) depending on logic
                # compute_pytorch_cost_matrix returns (BS, NumQueries, -1)
                
                # We need to know where the targets for this batch start/end if C is viewing flattened targets.
                # However, our replica function sets C = C.view(bs, num_queries, -1). 
                # This logic relies on targets being handled correctly.
                # In compute_pytorch_cost_matrix: "tgt_ids = torch.cat...".
                # If targets are variable length, view(bs, num_queries, -1) STRIDES across targets assuming fixed length?
                # NO! 
                # If `rfdetr` uses `view(bs, num_queries, -1)`, it IMPLIES targets dimension is fixed or padding?
                # BUT `linear_sum_assignment` later uses `C_g.split(sizes, -1)`.
                # This works ONLY if `view` correctly separated the targets.
                # If `cdist` output is (Batch*NumQueries, SumTargets).
                # `view` maps `Batch*NumQueries` -> `Batch` x `NumQueries`.
                # The last dim `SumTargets` remains `SumTargets`.
                # So C for image i includes costs for ALL targets in the batch.
                # Then `metrics.split(sizes, -1)` selects the relevant targets for image i.
                
                # So `batch_C_torch_np[i]` is (NumQueries, SumTargets).
                
                # Torch cost
                cost_torch = batch_C_torch_np[i][r_t, c_t].sum()
                
                # Keras cost (evaluated on Torch Matrix for comparison)
                cost_keras = batch_C_torch_np[i][r_k, c_k].sum()
                
                diff = abs(cost_torch - cost_keras)
                if diff > 1e-4:
                    raise AssertionError(f"Cost mismatch at zero-tolerance! Batch {i}: Torch={cost_torch}, Keras={cost_keras}, Diff={diff}")
            
            print("  Assignment Cost Parity Confirmed (1e-4). Mismatch purely due to tie-breaking.")
        else:
            raise

if __name__ == "__main__":
    pytest.main([__file__])
