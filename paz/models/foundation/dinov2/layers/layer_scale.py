import keras
import keras.ops as ops


def compute_layer_scale(x, gamma):
    return ops.multiply(x, gamma)


def LayerScale(
    dimension,
    init_values=1e-5,
    data_type=None,
    **kwargs,
):
    dtype = data_type or "float32"
    init = keras.initializers.Constant(init_values)
    gamma = keras.Variable(
        init(shape=(dimension,), dtype=dtype),
        trainable=True,
        dtype=dtype,
        name="gamma",
    )

    x_in = keras.Input(shape=(None, dimension))
    x_out = compute_layer_scale(x_in, init(shape=(dimension,), dtype=dtype))

    def call(x, training=None, **_):
        return compute_layer_scale(x, gamma)

    model = keras.Model(inputs=x_in, outputs=x_out, **kwargs)
    model.gamma = gamma
    model.call = call
    return model
