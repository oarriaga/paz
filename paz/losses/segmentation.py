from keras.saving import register_keras_serializable
from keras import ops

EPSILON = 1e-5


@register_keras_serializable("segmentation", "dice")
def dice(y_true, y_pred, beta=1.0):
    true_positives = ops.sum(y_true * y_pred, axis=[1, 2])
    false_positives = ops.sum(y_pred, axis=[1, 2]) - true_positives
    false_negatives = ops.sum(y_true, axis=[1, 2]) - true_positives
    squared_beta = beta**2
    numerator = (1.0 + squared_beta) * true_positives
    denominator = numerator + squared_beta * false_negatives + false_positives
    score = numerator / (denominator + EPSILON)
    return 1.0 - ops.mean(score, axis=-1)


@register_keras_serializable("segmentation", "jaccard")
def jaccard(y_true, y_pred):
    intersection = ops.sum(y_true * y_pred, axis=[1, 2])
    union = ops.sum(y_true + y_pred, axis=[1, 2]) - intersection
    score = intersection / (union + EPSILON)
    return 1.0 - ops.mean(score, axis=-1)


@register_keras_serializable("segmentation", "focal")
def focal(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = ops.clip(y_pred, EPSILON, 1.0 - EPSILON)
    modulator = alpha * ops.power(1.0 - y_pred, gamma)
    loss = -modulator * y_true * ops.log(y_pred)
    return ops.mean(ops.sum(loss, axis=-1), axis=[1, 2])
