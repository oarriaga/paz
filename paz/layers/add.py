import keras


class Add(keras.layers.Layer):
    """Wraps tensorflow's `add` function into a keras layer."""

    def __init__(self):
        super(Add, self).__init__()

    def call(self, x, y):
        return keras.ops.add(x, y)
