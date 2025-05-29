import keras
from keras import ops
from typing import Optional


def drop_path_keras_functional(
    x: keras.KerasTensor, drop_prob: float = 0.0, training: bool = False
) -> keras.KerasTensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob

    shape_x = ops.shape(x)
    shape_for_random = [shape_x[0]] + [1] * (len(shape_x) - 1)

    random_tensor = ops.random.bernoulli(shape_for_random, p=keep_prob, dtype=x.dtype)

    if keep_prob > 0.0:
        random_tensor = random_tensor / keep_prob

    output = x * random_tensor
    return output


class DropPath(keras.layers.Layer):
    def __init__(self, drop_prob: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        is_training_mode = training if training is not None else False

        return drop_path_keras_functional(x, self.drop_prob, is_training_mode)

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config
