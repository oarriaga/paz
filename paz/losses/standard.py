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


bce = binary_cross_entropy
mse = mean_squared_error
mae = mean_absolute_error


def weight(losses, weights):
    return (jp.array(losses) * jp.array(weights)).sum()
