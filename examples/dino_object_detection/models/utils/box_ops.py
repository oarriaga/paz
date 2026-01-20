import keras
from keras import ops


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_cxcywh_to_xyxy(x):
    x_c = x[..., 0]
    y_c = x[..., 1]
    w = x[..., 2]
    h = x[..., 3]

    w_half = 0.5 * ops.maximum(w, 0.0)
    h_half = 0.5 * ops.maximum(h, 0.0)

    b = [x_c - w_half, y_c - h_half, x_c + w_half, y_c + h_half]
    return ops.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0 = x[..., 0]
    y0 = x[..., 1]
    x1 = x[..., 2]
    y1 = x[..., 3]

    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return ops.stack(b, axis=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # boxes1: [N, 4] -> [N, 1, 2] for broadcasting
    # boxes2: [M, 4] -> [1, M, 2]
    lt = ops.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = ops.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = ops.maximum(rb - lt, 0.0)  # [N,M,2]
    inter = wh[..., 0] * wh[..., 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    # Avoid division by zero if needed, though PT impl doesn't check explicitly
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = ops.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = ops.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = ops.maximum(rb - lt, 0.0)  # [N,M,2]
    area = wh[..., 0] * wh[..., 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if ops.size(masks) == 0:
        return ops.zeros((0, 4))

    shape = ops.shape(masks)
    h, w = shape[-2], shape[-1]

    y = ops.arange(0, h, dtype="float32")
    x = ops.arange(0, w, dtype="float32")

    # indexing='ij' ensures y corresponds to rows (H) and x to cols (W)
    y, x = ops.meshgrid(y, x, indexing="ij")

    # Expand dims to broadcast against N
    x_expand = ops.expand_dims(x, axis=0)  # [1, H, W]
    y_expand = ops.expand_dims(y, axis=0)  # [1, H, W]

    x_mask = masks * x_expand
    y_mask = masks * y_expand

    # Flatten spatial dimensions: [N, H*W]
    x_mask_flat = ops.reshape(x_mask, (shape[0], -1))
    y_mask_flat = ops.reshape(y_mask, (shape[0], -1))

    x_max = ops.max(x_mask_flat, axis=-1)
    y_max = ops.max(y_mask_flat, axis=-1)

    # Replicating masked_fill logic using ops.where
    # masked_fill(~masks, 1e8) -> replace 0s with 1e8
    masks_bool = ops.cast(masks, "bool")
    masks_flat_bool = ops.reshape(masks_bool, (shape[0], -1))

    large_val = 1e8
    x_min_masked = ops.where(masks_flat_bool, x_mask_flat, large_val)
    y_min_masked = ops.where(masks_flat_bool, y_mask_flat, large_val)

    x_min = ops.min(x_min_masked, axis=-1)
    y_min = ops.min(y_min_masked, axis=-1)

    return ops.stack([x_min, y_min, x_max, y_max], axis=1)


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
    inputs = ops.sigmoid(inputs)

    # Flatten from dim 1: [N, ...] -> [N, Features]
    shape = ops.shape(inputs)
    inputs = ops.reshape(inputs, (shape[0], -1))

    # Note: Logic assumes targets is already flattened or compatible with einsum "mc"
    # If targets comes in as [N, H, W], it implies we should flatten it to match inputs logic
    if len(ops.shape(targets)) > 2:
        targets = ops.reshape(targets, (ops.shape(targets)[0], -1))

    numerator = 2 * ops.einsum("nc,mc->nm", inputs, targets)

    den_in = ops.sum(inputs, axis=-1)[:, None]
    den_tar = ops.sum(targets, axis=-1)[None, :]
    denominator = den_in + den_tar

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
    shape = ops.shape(inputs)
    # hw in PyTorch implementation refers to the feature dimension size (or H*W if flattened)
    hw = ops.cast(shape[1], "float32")

    # Keras ops.binary_crossentropy returns shape equal to inputs (element-wise)
    # when from_logits=True
    pos = ops.binary_crossentropy(ops.ones_like(inputs), inputs, from_logits=True)
    neg = ops.binary_crossentropy(ops.zeros_like(inputs), inputs, from_logits=True)

    # Flatten inputs/pos/neg if necessary for the einsum "nc,mc->nm"
    # The PyTorch einsum implies inputs are rank 2 [N, C] here.
    if len(shape) > 2:
        inputs = ops.reshape(inputs, (shape[0], -1))
        pos = ops.reshape(pos, (shape[0], -1))
        neg = ops.reshape(neg, (shape[0], -1))
        targets = ops.reshape(targets, (ops.shape(targets)[0], -1))
        # Recalculate hw based on flattened feature dim to match original logic
        hw = ops.cast(ops.shape(inputs)[1], "float32")

    loss = ops.einsum("nc,mc->nm", pos, targets) + ops.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw
