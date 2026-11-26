import keras


@keras.saving.register_keras_serializable("layers")
class Add(keras.layers.Layer):
    """Wraps tensorflow's `add` function into a keras layer."""

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, x, y):
        return keras.ops.add(x, y)
