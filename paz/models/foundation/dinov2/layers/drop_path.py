import keras
from keras import ops


def bernoulli_random(shape, probabilities, data_type=None):
    return keras.random.binomial(
        shape=shape, counts=1.0, probabilities=probabilities, dtype=data_type
    )


def dropout_rate(x, drop_probability=0.0, training=False):
    if drop_probability == 0.0 or not training:
        return x
    keep_probability = 1.0 - drop_probability
    shape_x = ops.shape(x)
    shape_for_random = [shape_x[0]] + [1] * (len(shape_x) - 1)
    random_tensor = bernoulli_random(
        shape=shape_for_random,
        probabilities=keep_probability,
        data_type=x.dtype,
    )
    if keep_probability > 0.0:
        random_tensor = random_tensor / keep_probability
    return x * random_tensor


def DropPath(drop_probability, **kwargs):
    probability = drop_probability if drop_probability is not None else 0.0
    x_in = keras.Input(shape=(None, None))
    x_out = ops.multiply(x_in, ops.ones(1))
    model = keras.Model(inputs=x_in, outputs=x_out, **kwargs)

    def call(x, training=None, **_):
        is_training = training if training is not None else False
        return dropout_rate(x, probability, is_training)

    model.call = call
    return model
