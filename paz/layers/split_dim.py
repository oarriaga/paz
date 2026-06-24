import keras
from keras import ops


@keras.saving.register_keras_serializable("layers")
class SplitDim(keras.layers.Layer):
    """Splits one dim into two adjacent dims at call time."""

    def __init__(self, axis, sizes, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.sizes = tuple(sizes)

    def get_config(self):
        config = super().get_config()
        config["axis"] = self.axis
        config["sizes"] = self.sizes
        return config

    def call(self, x):
        shape = ops.shape(x)
        idx = self.axis
        if idx < 0:
            idx = len(x.shape) + idx
        new_shape = shape[:idx] + self.sizes + shape[idx + 1:]
        return ops.reshape(x, new_shape)

    def compute_output_shape(self, input_shape):
        idx = self.axis
        if idx < 0:
            idx = len(input_shape) + idx
        return input_shape[:idx] + self.sizes + input_shape[idx + 1:]
