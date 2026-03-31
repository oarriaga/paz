import jax.numpy as jp


def _reduce_loss(loss, axis=None, reduction="mean"):
    if reduction == "mean":
        result = jp.mean(loss, axis=axis)
    elif reduction == "sum":
        result = jp.sum(loss, axis=axis)
    elif reduction == "none":
        result = loss
    else:
        raise ValueError("Unknown reduction. Expected: 'mean', 'sum', 'none'.")
    return result


def binary_cross_entropy(
    y_true, y_pred, axis=None, reduction="mean", epsilon=1e-7
):
    y_pred = jp.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -(y_true * jp.log(y_pred) + (1.0 - y_true) * jp.log(1.0 - y_pred))
    result = _reduce_loss(loss, axis=axis, reduction=reduction)
    return result


def mean_squared_error(y_true, y_pred, axis=None, reduction="mean"):
    loss = (y_true - y_pred) ** 2
    result = _reduce_loss(loss, axis=axis, reduction=reduction)
    return result


def mean_absolute_error(y_true, y_pred, axis=None, reduction="mean"):
    loss = jp.abs(y_true - y_pred)
    result = _reduce_loss(loss, axis=axis, reduction=reduction)
    return result


def _expand_masks(y_true, masks):
    H, W = y_true.shape[:2]
    masks = jp.array(masks)
    if masks.ndim == 2:
        masks = masks[None, ..., None]
    elif masks.ndim == 3:
        if masks.shape[:2] == (H, W):
            masks = masks[None]
        else:
            masks = masks[..., None]
    return masks.astype(bool)


def masked(loss_fn, y_true, y_pred, masks, mask_value=0.0):
    y_true = jp.array(y_true)
    y_pred = jp.array(y_pred)
    masks = _expand_masks(y_true, masks)
    masked_true = jp.where(masks, y_true[None], mask_value)
    masked_pred = jp.where(masks, y_pred[None], mask_value)
    loss = loss_fn(masked_true, masked_pred, reduction="none")
    axis = tuple(range(1, loss.ndim))
    return loss.mean(axis=axis).sum()


def masked_mean_absolute_error(y_true, y_pred, masks, mask_value=0.0):
    return masked(mean_absolute_error, y_true, y_pred, masks, mask_value)


def masked_mean_squared_error(y_true, y_pred, masks, mask_value=0.0):
    return masked(mean_squared_error, y_true, y_pred, masks, mask_value)


def masked_binary_cross_entropy(y_true, y_pred, masks, mask_value=0.0):
    return masked(binary_cross_entropy, y_true, y_pred, masks, mask_value)


def soft_box_barrier(
    values, min_val, max_val, curvature, axis=None, reduction="sum"
):
    negative = curvature * (values - min_val)
    positive = curvature * (values - max_val)
    loss = jp.exp(-negative) + jp.exp(positive)
    return _reduce_loss(loss, axis=axis, reduction=reduction)


bce = binary_cross_entropy
mse = mean_squared_error
mae = mean_absolute_error
masked_bce = masked_binary_cross_entropy
masked_mse = masked_mean_squared_error
masked_mae = masked_mean_absolute_error


def weight(losses, weights):
    return (jp.array(losses) * jp.array(weights)).sum()
