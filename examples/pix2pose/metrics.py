from loss import compute_weighted_reconstruction_loss_with_error
from loss import compute_error_prediction_loss
from loss import compute_weighted_reconstruction_loss


def weighted_reconstruction(RGBA_true, RGBE_pred, beta=3.0, with_error=False):
    if with_error:
        loss_function = compute_weighted_reconstruction_loss_with_error(RGBA_true, RGBE_pred, beta)
    else:
        loss_function = compute_weighted_reconstruction_loss(RGBA_true, RGBE_pred, beta)
    return loss_function


def error_prediction(RGBA_true, RGBE_pred, beta=3.0):
    return compute_error_prediction_loss(RGBA_true, RGBE_pred)
