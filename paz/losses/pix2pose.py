from keras.saving import register_keras_serializable
from keras import ops


@register_keras_serializable("pix2pose", "weighted_reconstruction")
def weighted_reconstruction(RGBA_true, RGB_pred, beta=3.0):
    RGB_true, alpha = RGBA_true[..., :3], RGBA_true[..., 3:4]
    error = ops.abs(RGB_true - RGB_pred)
    foreground = ops.mean(error * alpha, axis=-1)
    background = ops.mean(error * (1.0 - alpha), axis=-1)
    return beta * foreground + background


def WeightedReconstruction(beta=3.0):
    def loss(RGBA_true, RGB_pred):
        return weighted_reconstruction(RGBA_true, RGB_pred, beta)

    return loss
