import keras.ops as k


def box_cxcywh_to_xyxy(x):
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
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = k.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = k.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    # Clamp to zero so non-overlapping pairs get zero intersection
    wh = k.maximum(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    # Epsilon guards division by zero for degenerate boxes
    intersection_over_union = inter / (union + 1e-6)
    return intersection_over_union, union


def generalized_box_iou(boxes1, boxes2):
    intersection_over_union, union = box_iou(boxes1, boxes2)
    lt = k.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = k.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = k.maximum(rb - lt, 0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return intersection_over_union - (area - union) / (area + 1e-6)


def masks_to_boxes(masks):
    if k.shape(masks)[0] == 0:
        boxes = k.zeros((0, 4))
    else:
        mask_shape = k.shape(masks)
        h, w = mask_shape[-2], mask_shape[-1]
        y = k.arange(0, h, dtype="float32")
        x = k.arange(0, w, dtype="float32")
        y_grid = k.expand_dims(y, axis=1) * k.ones((1, w), dtype="float32")
        x_grid = k.ones((h, 1), dtype="float32") * k.expand_dims(x, axis=0)
        x_mask = masks * k.expand_dims(x_grid, 0)
        x_max = k.max(k.reshape(x_mask, (mask_shape[0], -1)), axis=-1)
        # Fill non-mask pixels with a large value so min ignores them
        inv_masks_bool = k.logical_not(k.cast(masks, "bool"))
        x_mask_filled = k.where(inv_masks_bool, 1e8, x_mask)
        x_min = k.min(k.reshape(x_mask_filled, (mask_shape[0], -1)), axis=-1)
        y_mask = masks * k.expand_dims(y_grid, 0)
        y_max = k.max(k.reshape(y_mask, (mask_shape[0], -1)), axis=-1)
        y_mask_filled = k.where(inv_masks_bool, 1e8, y_mask)
        y_min = k.min(k.reshape(y_mask_filled, (mask_shape[0], -1)), axis=-1)
        boxes = k.stack([x_min, y_min, x_max, y_max], axis=1)
    return boxes


def batch_dice_loss(inputs, targets):
    inputs = k.sigmoid(inputs)
    inputs = k.reshape(inputs, (k.shape(inputs)[0], -1))
    targets = k.cast(targets, inputs.dtype)
    targets = k.reshape(targets, (k.shape(targets)[0], -1))
    # Pairwise dot product gives the per-pair DICE-numerator intersection
    numerator = 2 * k.matmul(inputs, k.transpose(targets, (1, 0)))
    inputs_sum = k.sum(inputs, axis=-1)[:, None]
    targets_sum = k.sum(targets, axis=-1)[None, :]
    denominator = inputs_sum + targets_sum
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs, targets):
    hw = k.shape(inputs)[1]
    # Per-element BCE against all-ones and all-zeros, combined by the targets
    pos = k.binary_crossentropy(k.ones_like(inputs), inputs, from_logits=True)
    neg = k.binary_crossentropy(k.zeros_like(inputs), inputs, from_logits=True)
    pos_flat = k.reshape(pos, (k.shape(pos)[0], -1))
    neg_flat = k.reshape(neg, (k.shape(neg)[0], -1))
    targets_2d = k.reshape(targets, (k.shape(targets)[0], -1))
    targets_flat = k.cast(targets_2d, inputs.dtype)
    # Weight positive-class BCE by target and negative by (1 - target)
    term1 = k.matmul(pos_flat, k.transpose(targets_flat, (1, 0)))
    term2 = k.matmul(neg_flat, k.transpose(1 - targets_flat, (1, 0)))
    loss = term1 + term2
    return loss / k.cast(hw, "float32")
