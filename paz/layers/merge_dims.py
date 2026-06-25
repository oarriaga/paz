import keras
from keras import ops


@keras.saving.register_keras_serializable("layers")
class MergeDims(keras.layers.Layer):
    """Merges two adjacent dims at call time (no parameters)."""

    def __init__(self, axis=-2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        return config

    def call(self, x):
        shape = ops.shape(x)
        idx = self.axis
        if idx < 0:
            idx = len(x.shape) + idx
        merged = shape[idx] * shape[idx + 1]
        new_shape = shape[:idx] + (merged,) + shape[idx + 2:]
        return ops.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        idx = self.axis
        if idx < 0:
            idx = len(input_shape) + idx
        a, b = input_shape[idx], input_shape[idx + 1]
        merged = None if (a is None or b is None) else a * b
        return input_shape[:idx] + (merged,) + input_shape[idx + 2:]
