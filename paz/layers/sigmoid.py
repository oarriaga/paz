import keras


class Sigmoid(keras.layers.Layer):
    """Wraps tensorflow's `sigmoid` function into a keras layer."""

    def __init__(self):
        super(Sigmoid, self).__init__()

    def call(self, x):
        return keras.activations.sigmoid(x)
