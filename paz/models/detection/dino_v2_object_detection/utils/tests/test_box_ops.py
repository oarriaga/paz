import os
import sys
# Add parent directory to path to allow importing box_ops
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch
import keras
import keras.ops as k
import box_ops as keras_box_ops

# Original PyTorch implementations for reference
# Taken from original file
def pt_box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def pt_box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def pt_box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def pt_generalized_box_iou(boxes1, boxes2):
    iou, union = pt_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def pt_masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x) # indexing='ij' by default in recent torch

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# Helper function for verification
def verify_output(keras_out, torch_out, atol=1e-5):
    k_out = keras.ops.convert_to_numpy(keras_out)
    t_out = torch_out.detach().cpu().numpy()
    np.testing.assert_allclose(k_out, t_out, atol=atol, err_msg="Outputs do not match")

def test_box_cxcywh_to_xyxy():
    # Random inputs
    x_cxcywh = np.random.rand(10, 4).astype(np.float32)
    # Ensure w, h are positive
    x_cxcywh[:, 2:] = np.abs(x_cxcywh[:, 2:])
    
    # Run Keras
    k_in = keras.ops.convert_to_tensor(x_cxcywh)
    k_out = keras_box_ops.box_cxcywh_to_xyxy(k_in)
    
    # Run Torch
    t_in = torch.tensor(x_cxcywh)
    t_out = pt_box_cxcywh_to_xyxy(t_in)
    
    verify_output(k_out, t_out)

def test_box_xyxy_to_cxcywh():
    # Random inputs
    x_xyxy = np.random.rand(10, 4).astype(np.float32)
    # Ensure x2 > x1, y2 > y1
    x_xyxy[:, 2] = x_xyxy[:, 0] + np.abs(x_xyxy[:, 2])
    x_xyxy[:, 3] = x_xyxy[:, 1] + np.abs(x_xyxy[:, 3])
    
    # Run Keras
    k_in = keras.ops.convert_to_tensor(x_xyxy)
    k_out = keras_box_ops.box_xyxy_to_cxcywh(k_in)
    
    # Run Torch
    t_in = torch.tensor(x_xyxy)
    t_out = pt_box_xyxy_to_cxcywh(t_in)
    
    verify_output(k_out, t_out)

def test_box_iou():
    # Boxes 1
    b1 = np.random.rand(10, 4).astype(np.float32)
    b1[:, 2] = b1[:, 0] + np.abs(b1[:, 2])
    b1[:, 3] = b1[:, 1] + np.abs(b1[:, 3])
    
    # Boxes 2
    b2 = np.random.rand(5, 4).astype(np.float32)
    b2[:, 2] = b2[:, 0] + np.abs(b2[:, 2])
    b2[:, 3] = b2[:, 1] + np.abs(b2[:, 3])
    
    # Run Keras
    k_b1 = keras.ops.convert_to_tensor(b1)
    k_b2 = keras.ops.convert_to_tensor(b2)
    k_iou, k_union = keras_box_ops.box_iou(k_b1, k_b2)
    
    # Run Torch
    t_b1 = torch.tensor(b1)
    t_b2 = torch.tensor(b2)
    t_iou, t_union = pt_box_iou(t_b1, t_b2)
    
    verify_output(k_iou, t_iou, atol=1e-5)
    verify_output(k_union, t_union, atol=1e-5)

def test_generalized_box_iou():
    # Boxes 1
    b1 = np.random.rand(10, 4).astype(np.float32)
    b1[:, 2] = b1[:, 0] + np.abs(b1[:, 2])
    b1[:, 3] = b1[:, 1] + np.abs(b1[:, 3])
    
    # Boxes 2
    b2 = np.random.rand(5, 4).astype(np.float32)
    b2[:, 2] = b2[:, 0] + np.abs(b2[:, 2])
    b2[:, 3] = b2[:, 1] + np.abs(b2[:, 3])
    
    # Run Keras
    k_b1 = keras.ops.convert_to_tensor(b1)
    k_b2 = keras.ops.convert_to_tensor(b2)
    k_giou = keras_box_ops.generalized_box_iou(k_b1, k_b2)
    
    # Run Torch
    t_b1 = torch.tensor(b1)
    t_b2 = torch.tensor(b2)
    t_giou = pt_generalized_box_iou(t_b1, t_b2)
    
    verify_output(k_giou, t_giou, atol=1e-5)

def test_masks_to_boxes():
    # Masks (N, H, W)
    masks = np.random.randint(0, 2, size=(3, 10, 10)).astype(np.float32)
    
    # Ensure at least one pixel is 1 in each mask to avoid infs in naive implementation
    masks[:, 5, 5] = 1.0 
    
    # Run Keras
    k_in = keras.ops.convert_to_tensor(masks)
    k_out = keras_box_ops.masks_to_boxes(k_in)
    
    # Run Torch
    t_in = torch.tensor(masks)
    t_out = pt_masks_to_boxes(t_in)
    
    verify_output(k_out, t_out)

def test_masks_to_boxes_empty():
    masks = np.zeros((0, 10, 10)).astype(np.float32)
    
    k_in = keras.ops.convert_to_tensor(masks)
    k_out = keras_box_ops.masks_to_boxes(k_in)
    
    assert keras.ops.shape(k_out)[0] == 0
    
    t_in = torch.tensor(masks)
    t_out = pt_masks_to_boxes(t_in)
    
    # Torch returns (0, 4)
    assert t_out.shape == (0, 4)
