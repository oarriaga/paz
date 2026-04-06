import jax
import jax.numpy as jp
import paz

from paz.losses.standard import _reduce_loss


def _add_batch_axis(values):
    values = jp.array(values)
    if values.ndim == 3:
        values = jp.expand_dims(values, axis=0)
    return values


def _guided_smoothing(true_depth, pred_depth):
    dy_true, dx_true = paz.image.forward_differences(true_depth)
    dy_pred, dx_pred = paz.image.forward_differences(pred_depth)
    x_weight = jp.exp(jp.mean(jp.abs(dx_true)))
    y_weight = jp.exp(jp.mean(jp.abs(dy_true)))
    x_smooth = jp.mean(jp.abs(dx_pred * x_weight))
    y_smooth = jp.mean(jp.abs(dy_pred * y_weight))
    return x_smooth + y_smooth


def guided_smoothing(true_depth, pred_depth, reduction="mean"):
    """Shapes: true_depth and pred_depth are (H, W, C) or (B, H, W, C)."""
    true_depth = _add_batch_axis(true_depth)
    pred_depth = _add_batch_axis(pred_depth)
    loss = jax.vmap(_guided_smoothing)(true_depth, pred_depth)
    return _reduce_loss(loss, reduction=reduction)
