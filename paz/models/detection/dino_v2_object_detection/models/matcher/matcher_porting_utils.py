import os
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    import keras
    from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher as KerasHungarianMatcher
except ImportError:
    try:
        from paz.models.detection.dino_v2_object_detection.models.matcher.matcher import HungarianMatcher as KerasHungarianMatcher
    except ImportError:
         print("Warning: Could not import KerasHungarianMatcher. Ensure project root is in python path.")
         KerasHungarianMatcher = None

def to_numpy(t):
    """Converts a tensor to a numpy array.

    Handles torch tensors, keras tensors, and numpy arrays uniformly.

    Args:
        t: Input tensor (torch.Tensor, keras tensor, or np.ndarray).

    Returns:
        np.ndarray: Numpy array representation of the input.
    """
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.array(t)

def convert_to_keras(outputs_torch, targets_torch):
    """Converts model outputs and targets to numpy-backed format.

    Transforms torch tensor outputs and target dictionaries into numpy
    arrays suitable for use with the Keras HungarianMatcher.

    Args:
        outputs_torch (dict): Model predictions with torch tensor values.
            Expected keys: "pred_logits", "pred_boxes", optional
            "pred_masks" (tensor or dict of tensors).
        targets_torch (list[dict]): Per-image target dicts with torch
            tensor values. Expected keys: "labels", "boxes", optional
            "masks".

    Returns:
        tuple: (outputs_numpy, targets_numpy) with the same structure
            but numpy array values.
    """
    outputs_keras = {}
    # Convert prediction tensors to numpy
    outputs_keras["pred_logits"] = to_numpy(outputs_torch["pred_logits"])
    outputs_keras["pred_boxes"] = to_numpy(outputs_torch["pred_boxes"])
    
    # Handle mask predictions: either a single tensor or a dictionary
    # of component tensors for deferred mask computation
    if "pred_masks" in outputs_torch:
        if isinstance(outputs_torch["pred_masks"], torch.Tensor):
            outputs_keras["pred_masks"] = to_numpy(outputs_torch["pred_masks"])
        else:
            # Sparse/deferred masks: dictionary of component tensors
            outputs_keras["pred_masks"] = {}
            for k, v in outputs_torch["pred_masks"].items():
                outputs_keras["pred_masks"][k] = to_numpy(v)

    # Convert per-image target dictionaries to numpy
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
    """Extracts matcher configuration from model arguments.

    Reads cost weights and focal alpha from the model's argument object.
    If a segmentation head is present, also extracts mask cost parameters
    with sensible defaults.

    Args:
        args: Model configuration object with attributes set_cost_class,
            set_cost_bbox, set_cost_giou, focal_alpha, and optionally
            segmentation_head, mask_ce_loss_coef, mask_dice_loss_coef,
            mask_point_sample_ratio.

    Returns:
        dict: Configuration dictionary with keys cost_class, cost_bbox,
            cost_giou, focal_alpha, and optionally cost_mask_ce,
            cost_mask_dice, mask_point_sample_ratio.
    """
    config = {
        "cost_class": args.set_cost_class,
        "cost_bbox": args.set_cost_bbox,
        "cost_giou": args.set_cost_giou,
        "focal_alpha": args.focal_alpha,
    }
    
    if getattr(args, 'segmentation_head', False):
        # Inject default mask cost values when not explicitly configured
        config["cost_mask_ce"] = getattr(args, 'mask_ce_loss_coef', 5.0)
        config["cost_mask_dice"] = getattr(args, 'mask_dice_loss_coef', 5.0)
        config["mask_point_sample_ratio"] = getattr(args, 'mask_point_sample_ratio', 16)
        
    return config

def build_keras_matcher_from_config(config):
    """Builds a HungarianMatcher instance from a configuration dictionary.

    Args:
        config (dict): Matcher configuration with keys cost_class,
            cost_bbox, cost_giou, focal_alpha, and optionally
            mask_point_sample_ratio, cost_mask_ce, cost_mask_dice.

    Returns:
        HungarianMatcher: Configured matcher instance.

    Raises:
        ImportError: If KerasHungarianMatcher could not be imported.
    """
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
    """Asserts that two sets of matcher outputs are equivalent.

    Compares matched index pairs from two matcher implementations to
    verify they produce identical assignments.

    Args:
        indices_torch (list[tuple]): Reference matcher output as a list
            of (row_indices, col_indices) tuples.
        indices_keras (list[tuple]): Matcher output to validate, same
            format as indices_torch.
        check_exact (bool): If True, asserts exact index equality.
            If False, only checks shape consistency (useful when
            randomized mask sampling causes non-deterministic results).

    Raises:
        AssertionError: If batch sizes differ, shapes mismatch, or
            (when check_exact=True) index values differ.
    """
    assert len(indices_torch) == len(indices_keras), "Number of batch elements matched differs"
    
    for i in range(len(indices_torch)):
        ind_i_torch, ind_j_torch = indices_torch[i]
        ind_i_keras, ind_j_keras = indices_keras[i]
        
        # Convert reference indices to numpy for comparison
        ind_i_torch_np = to_numpy(ind_i_torch)
        ind_j_torch_np = to_numpy(ind_j_torch)
        
        assert ind_i_torch_np.shape == ind_i_keras.shape, f"Shape mismatch at batch index {i}"
        
        if check_exact:
            try:
                np.testing.assert_array_equal(ind_i_torch_np, ind_i_keras)
                np.testing.assert_array_equal(ind_j_torch_np, ind_j_keras)
            except AssertionError as e:
                raise AssertionError(f"Index mismatch at batch index {i}: {e}")
