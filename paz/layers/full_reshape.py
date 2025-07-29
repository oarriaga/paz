import keras
from keras import ops


@keras.saving.register_keras_serializable("layers")
class FullReshape(keras.Layer):
    """Reshapes all tensor dimensions including the batch dimension."""

    def __init__(self, shape, **kwargs):
        super(FullReshape, self).__init__(**kwargs)
        self.shape = shape

    def call(self, x):
        return ops.reshape(x, self.shape)

    def get_config(self):
        config = super().get_config()
        config.update({"shape": self.shape})
        return config
