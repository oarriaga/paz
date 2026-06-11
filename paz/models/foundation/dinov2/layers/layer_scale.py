import keras
from keras.layers import EinsumDense

from paz.models.foundation.dinov2.layers.drop_path import apply_identity


def apply_layer_scale(x, dim, init_values, name):
    if init_values:
        result = scale(x, dim, init_values, name)
    else:
        result = apply_identity(x, name)
    return result


def scale(x, dim, init_values, name):
    initializer = keras.initializers.Constant(init_values)
    # add comment her to explain what does this einsum layer do, and why it is used for layer scaling
    # The EinsumDense layer is used here to perform element-wise scaling
    # of the input tensor 'x' by a learnable parameter vector of size 'dim'.
    # The equation "...d,d->...d" indicates that for each element in the last dimension of 'x',
    # it will be multiplied by the corresponding element in the parameter vector.
    # This allows the model to learn how much to scale each feature in the input,
    # which can help with training stability and performance.
    # The initializer is set to a constant value specified by 'init_values',
    # which means that initially, the scaling will be uniform across all features,
    # but it can be adjusted during training.
    layer = EinsumDense(
        equation="...d,d->...d",
        output_shape=(dim,),
        bias_axes=None,
        kernel_initializer=initializer,
        name=name,
    )
    return layer(x)
