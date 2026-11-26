import keras


@keras.saving.register_keras_serializable("layers")
class Sigmoid(keras.layers.Layer):
    """Wraps tensorflow's `sigmoid` function into a keras layer."""

    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def call(self, x):
        return keras.activations.sigmoid(x)
