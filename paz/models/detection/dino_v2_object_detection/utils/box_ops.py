import keras
import keras.ops as k
from keras import random

def box_cxcywh_to_xyxy(x):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) is the center of the box, w is width, h is height.
    """
    x_c, y_c, w, h = k.split(x, 4, axis=-1)
    x_c = k.squeeze(x_c, axis=-1)
    y_c = k.squeeze(y_c, axis=-1)
    w = k.squeeze(w, axis=-1)
    h = k.squeeze(h, axis=-1)

    b = [
        (x_c - 0.5 * w),
        (y_c - 0.5 * h),
        (x_c + 0.5 * w),
        (y_c + 0.5 * h),
    ]
    return k.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    """
    x0, y0, x1, y1 = k.split(x, 4, axis=-1)
    x0 = k.squeeze(x0, axis=-1)
    y0 = k.squeeze(y0, axis=-1)
    x1 = k.squeeze(x1, axis=-1)
    y1 = k.squeeze(y1, axis=-1)

    b = [
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        (x1 - x0),
        (y1 - y0),
    ]
    return k.stack(b, axis=-1)


def box_iou(boxes1, boxes2):
    """
    Computes Intersection over Union (IoU) of two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1: (N, 4) tensor
        boxes2: (M, 4) tensor

    Returns:
        iou: (N, M) matrix containing the pairwise IoU values
        union: (N, M) matrix containing the pairwise union areas
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # boxes1[:, None, :2] shape: (N, 1, 2)
    # boxes2[:, :2] shape: (M, 2)
    # Result: (N, M, 2)
    lt = k.maximum(boxes1[:, None, :2], boxes2[:, :2])
    
    # boxes1[:, None, 2:] shape: (N, 1, 2)
    # boxes2[:, 2:] shape: (M, 2)
    # Result: (N, M, 2)
    rb = k.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = k.maximum(rb - lt, 0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6) # Add epsilon to avoid division by zero
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union = box_iou(boxes1, boxes2)

    lt = k.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = k.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = k.maximum(rb - lt, 0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if k.shape(masks)[0] == 0:
        return k.zeros((0, 4))

    mask_shape = k.shape(masks)
    h, w = mask_shape[-2], mask_shape[-1]
    
    # Cast to float for calculations
    h_float = k.cast(h, "float32")
    w_float = k.cast(w, "float32")

    y = k.arange(0, h, dtype="float32")
    x = k.arange(0, w, dtype="float32")
    
    # meshgrid in keras/numpy: indexing='xy' by default, 
    # but PyTorch (and this logic) expects 'ij' semantics or manual handling.
    # PyTorch: y, x = torch.meshgrid(y, x) -> y is (H, W) varying along dim 0, x is (H, W) varying along dim 1
    # Keras/Numpy: meshgrid(x, y, indexing='xy') -> x is (H, W) varying along 1, y is (H, W) varying along 0. 
    # Let's check:
    # y = [0, 1, ... H-1], x = [0, 1, ... W-1]
    # meshgrid(y, x, indexing='ij') -> y_grid[i, j] = i, x_grid[i, j] = j. This matches torch default? 
    # Actually torch default is 'ij' for recent versions, but old was 'strided' warning.
    # The original code:
    # y = torch.arange(0, h)
    # x = torch.arange(0, w)
    # y, x = torch.meshgrid(y, x) 
    # y_grid has shape (H, W), values are 0..H-1 per row
    # x_grid has shape (H, W), values are 0..W-1 per col
    
    x_grid, y_grid = k.meshgrid(x, y) # Default xy indexing: x_grid (H, W), y_grid (H, W)
    # Wait, numpy meshgrid(x, y) returns (H, W) arrays if indexing='xy'? 
    # No, meshgrid(x, y) -> x_grid (len(y), len(x)), y_grid (len(y), len(x))
    # Let's use explicit broadcasting to be safe and clear.
    
    y_grid = k.expand_dims(y, axis=1) * k.ones((1, w), dtype="float32") # (H, W)
    x_grid = k.ones((h, 1), dtype="float32") * k.expand_dims(x, axis=0) # (H, W)

    x_mask = (masks * k.expand_dims(x_grid, 0))
    x_max = k.max(k.reshape(x_mask, (mask_shape[0], -1)), axis=-1)
    
    # masked_fill logic
    # ~masks in pytorch. masks is 0 or 1 float here?
    # Original: masks * x.unsqueeze(0). masks is likely float or bool.
    # If masks is float 0.0/1.0, ~(masks.bool()) is True where mask is 0.
    
    masks_bool = k.cast(masks, "bool")
    inv_masks_bool = k.logical_not(masks_bool)
    
    # Fill with large value where mask is 0
    x_mask_filled = k.where(inv_masks_bool, 1e8, x_mask)
    x_min = k.min(k.reshape(x_mask_filled, (mask_shape[0], -1)), axis=-1)

    y_mask = (masks * k.expand_dims(y_grid, 0))
    y_max = k.max(k.reshape(y_mask, (mask_shape[0], -1)), axis=-1)
    
    y_mask_filled = k.where(inv_masks_bool, 1e8, y_mask)
    y_min = k.min(k.reshape(y_mask_filled, (mask_shape[0], -1)), axis=-1)

    return k.stack([x_min, y_min, x_max, y_max], axis=1)


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = k.sigmoid(inputs)
    inputs = k.reshape(inputs, (k.shape(inputs)[0], -1))
    targets = k.cast(targets, inputs.dtype) # Ensure targets are same type
    targets = k.reshape(targets, (k.shape(targets)[0], -1))

    # numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    # This einsum computes for every pair of (prediction_n, target_m). 
    # This looks like it computes an N x N matrix of dice losses?
    # Original code: 
    # numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    # This computes a matrix. Is this intended? 
    # Yes, usually used for hungarian matching cost computation.
    
    # Keras ops doesn't have native einsum yet? checking... 
    # It does have `numpy.einsum` equivalent via backend? 
    # Use k.matmul: (N, C) @ (M, C).T -> (N, M)
    numerator = 2 * k.matmul(inputs, k.transpose(targets, (1, 0)))
    
    denominator = k.sum(inputs, axis=-1)[:, None] + k.sum(targets, axis=-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs, targets):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = k.shape(inputs)[1]
    
    # F.binary_cross_entropy_with_logits(input, target, reduction='none')
    # = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # or just use k.binary_crossentropy(target, input, from_logits=True)
    
    # Note: Keras binary_crossentropy(y_true, y_pred, from_logits=True) expects (batch, ...)
    # But here we need to compute it against all ones and all zeros.
    
    # pos: BCE(inputs, ones)
    pos = k.binary_crossentropy(k.ones_like(inputs), inputs, from_logits=True)
    
    # neg: BCE(inputs, zeros)
    neg = k.binary_crossentropy(k.zeros_like(inputs), inputs, from_logits=True)

    # All flattened? No, original uses reduction='none', so shape preserved.
    # original: inputs is (N, C), targets is (M, C).
    # wait, the original logic:
    # pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    # pos shape: same as inputs
    
    # loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    
    # We flatten along C dimension for dot product
    pos_flat = k.reshape(pos, (k.shape(pos)[0], -1))
    neg_flat = k.reshape(neg, (k.shape(neg)[0], -1))
    
    targets_flat = k.cast(k.reshape(targets, (k.shape(targets)[0], -1)), inputs.dtype)
    
    term1 = k.matmul(pos_flat, k.transpose(targets_flat, (1, 0)))
    term2 = k.matmul(neg_flat, k.transpose(1 - targets_flat, (1, 0)))
    
    loss = term1 + term2
    return loss / k.cast(hw, "float32")
