import keras
from keras.layers import Dense, Dropout, Activation


def compute_mlp(x, fc1, activation, fc2, drop, training):
    x = fc1(x)
    x = activation(x)
    x = drop(x, training=training)
    x = fc2(x)
    x = drop(x, training=training)
    return x


def MLP(
    input_features,
    hidden_features=None,
    output_features=None,
    activation_layer=None,
    drop_rate=0.0,
    use_bias=True,
    **kwargs,
):
    hidden = hidden_features if hidden_features is not None else input_features
    out = output_features if output_features is not None else input_features
    init = keras.initializers.TruncatedNormal(stddev=0.02)
    kw = {"use_bias": use_bias, "kernel_initializer": init}
    fc1 = Dense(hidden, name="fully_connected_layer_1", **kw)
    fc2 = Dense(out, name="fully_connected_layer_2", **kw)
    activation = (
        activation_layer if activation_layer is not None else Activation("gelu")
    )
    drop = Dropout(drop_rate, name="drop")

    x_in = keras.Input(shape=(None, input_features))

    def call(x, training=None, **_):
        return compute_mlp(x, fc1, activation, fc2, drop, training)

    x_out = compute_mlp(x_in, fc1, activation, fc2, drop, None)
    model = keras.Model(inputs=x_in, outputs=x_out, **kwargs)
    model.fully_connected_layer_1 = fc1
    model.fully_connected_layer_2 = fc2
    model.activation = activation
    model.drop_layer = drop
    model.call = call
    return model
