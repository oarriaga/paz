import keras
from keras import ops
from typing import Optional


def bernoulli_random(shape, probabilities, dtype=None):
    """
    Generate random samples from a Bernoulli distribution.

    Args:
        shape: Shape of the output tensor
        p: Probability of success (0 <= p <= 1)
        dtype: Data type of the output tensor

    Returns:
        Tensor with random binary values (0 or 1)
    """
    return keras.random.binomial(shape=shape, counts=1.0, probabilities=probabilities, dtype=dtype)


def dropout_rate(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob

    shape_x = ops.shape(x)
    shape_for_random = [shape_x[0]] + [1] * (len(shape_x) - 1)

    random_tensor = bernoulli_random(shape=shape_for_random, probabilities=keep_prob, dtype=x.dtype)

    if keep_prob > 0.0:
        random_tensor = random_tensor / keep_prob

    output = x * random_tensor
    return output


class DropPath(keras.layers.Layer):
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def call(self, x, training=None):
        is_training_mode = training if training is not None else False

        return dropout_rate(x, self.drop_prob, is_training_mode)

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config
