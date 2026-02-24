import os
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Note: We assume Keras is available in the environment where this is run
try:
    import keras
    from examples.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher as KerasHungarianMatcher
except ImportError:
    # Try importing assuming run from root
    try:
        from examples.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher as KerasHungarianMatcher
    except ImportError:
         print("Warning: Could not import KerasHungarianMatcher. Ensure project root is in python path.")
         KerasHungarianMatcher = None

def to_numpy(t):
    """Converts a tensor (Torch/Keras/Numpy) to a numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"): # Keras tensor
        return t.numpy()
    return np.array(t)

def convert_to_keras(outputs_torch, targets_torch):
    """
    Convert PyTorch outputs and targets to the format expected by Keras matcher.
    """
    outputs_keras = {}
    outputs_keras["pred_logits"] = to_numpy(outputs_torch["pred_logits"])
    outputs_keras["pred_boxes"] = to_numpy(outputs_torch["pred_boxes"])
    
    if "pred_masks" in outputs_torch:
        if isinstance(outputs_torch["pred_masks"], torch.Tensor):
            outputs_keras["pred_masks"] = to_numpy(outputs_torch["pred_masks"])
        else:
            # Sparse masks (dictionary)
            outputs_keras["pred_masks"] = {}
            for k, v in outputs_torch["pred_masks"].items():
                outputs_keras["pred_masks"][k] = to_numpy(v)

    targets_keras = []
    for t in targets_torch:
        t_keras = {}
        t_keras["labels"] = to_numpy(t["labels"])
        t_keras["boxes"] = to_numpy(t["boxes"])
        if "masks" in t:
             t_keras["masks"] = to_numpy(t["masks"])
        targets_keras.append(t_keras)
        
    return outputs_keras, targets_keras

def extract_matcher_config(args):
    """
    Extracts matcher configuration from RF-DETR args object.
    Injects default values for mask coefficients if segmentation head is present but args are missing.
    """
    config = {
        "cost_class": args.set_cost_class,
        "cost_bbox": args.set_cost_bbox,
        "cost_giou": args.set_cost_giou,
        "focal_alpha": args.focal_alpha,
    }
    
    if getattr(args, 'segmentation_head', False):
        # Default values if missing (e.g. inference mode config)
        config["cost_mask_ce"] = getattr(args, 'mask_ce_loss_coef', 5.0)
        config["cost_mask_dice"] = getattr(args, 'mask_dice_loss_coef', 5.0)
        config["mask_point_sample_ratio"] = getattr(args, 'mask_point_sample_ratio', 16)
        
    return config

def build_keras_matcher_from_config(config):
    """Builds a Keras HungarianMatcher from a configuration dictionary."""
    if KerasHungarianMatcher is None:
        raise ImportError("KerasHungarianMatcher not imported.")
        
    return KerasHungarianMatcher(
        cost_class=config["cost_class"],
        cost_bbox=config["cost_bbox"],
        cost_giou=config["cost_giou"],
        focal_alpha=config["focal_alpha"],
        mask_point_sample_ratio=config.get("mask_point_sample_ratio", 16),
        cost_mask_ce=config.get("cost_mask_ce", 1.0),
        cost_mask_dice=config.get("cost_mask_dice", 1.0)
    )

def assert_matcher_parity(indices_torch, indices_keras, check_exact=True):
    """
    Asserts parity between PyTorch and Keras matcher outputs.
    indices_torch: List of (row_ind, col_ind) tensors from PyTorch matcher.
    indices_keras: List of (row_ind, col_ind) tensors/arrays from Keras matcher.
    check_exact: If False, skips exact index comparison (useful for randomized sampling cases like masks).
    """
    assert len(indices_torch) == len(indices_keras), "Number of batch elements matched differs"
    
    for i in range(len(indices_torch)):
        ind_i_torch, ind_j_torch = indices_torch[i]
        ind_i_keras, ind_j_keras = indices_keras[i]
        
        ind_i_torch_np = to_numpy(ind_i_torch)
        ind_j_torch_np = to_numpy(ind_j_torch)
        
        # Check shapes
        assert ind_i_torch_np.shape == ind_i_keras.shape, f"Shape mismatch at batch index {i}"
        
        if check_exact:
            try:
                np.testing.assert_array_equal(ind_i_torch_np, ind_i_keras)
                np.testing.assert_array_equal(ind_j_torch_np, ind_j_keras)
            except AssertionError as e:
                raise AssertionError(f"Index mismatch at batch index {i}: {e}")
