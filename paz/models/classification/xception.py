from keras import Model
from keras.regularizers import l2
from keras.utils import get_file
from keras import layers
from keras.layers import (
    Conv2D,
    BatchNormalization,
    SeparableConv2D,
    Activation,
    MaxPooling2D,
    Add,
    Input,
    GlobalAveragePooling2D,
    Rescaling,
)


def xception_block(input_tensor, num_kernels, l2_reg=0.01):
    """Xception core block.

    # Arguments
        input_tenso: Keras tensor.
        num_kernels: Int. Number of convolutional kernels in block.
        l2_reg: Float. l2 regression.

    # Returns
        output tensor for the block.
    """
    residual = Conv2D(
        num_kernels, 1, strides=(2, 2), padding="same", use_bias=False
    )(input_tensor)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        num_kernels,
        3,
        padding="same",
        depthwise_regularizer=l2(l2_reg),
        pointwise_regularizer=l2(l2_reg),
        use_bias=False,
    )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        num_kernels,
        3,
        padding="same",
        depthwise_regularizer=l2(l2_reg),
        pointwise_regularizer=l2(l2_reg),
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=(2, 2), padding="same")(x)
    x = Add()([x, residual])
    return x


def build_xception(
    input_shape,
    num_classes,
    stem_kernels,
    block_kernels,
    classifier_activation="softmax",
    l2_reg=0.01,
):
    """Function for instantiating an Xception model.

    # Arguments
        input_shape: List corresponding to the input shape of the model.
        num_classes: Integer.
        stem_kernels: List of integers. Each element of the list indicates
            the number of kernels used as stem blocks.
        block_kernels: List of integers. Each element of the list Indicates
            the number of kernels used in the xception blocks.
        l2_reg. Float. L2 regularization used in the convolutional kernels.

    # Returns
        Tensorflow-Keras model.

    # References
        - [Xception: Deep Learning with Depthwise Separable
            Convolutions](https://arxiv.org/abs/1610.02357)
    """

    x = inputs = Input(input_shape, name="image")
    for num_kernels in stem_kernels:
        x = Conv2D(
            num_kernels,
            3,
            kernel_regularizer=l2(l2_reg),
            use_bias=False,
            padding="same",
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    for num_kernels in block_kernels:
        x = xception_block(x, num_kernels, l2_reg)

    x = Conv2D(num_classes, 3, kernel_regularizer=l2(l2_reg), padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation(classifier_activation, name="label")(x)

    model_name = "-".join(
        [
            "XCEPTION",
            str(input_shape[0]),
            str(stem_kernels[0]),
            str(len(block_kernels)),
        ]
    )
    model = Model(inputs, output, name=model_name)
    return model


def MiniXception(
    input_shape,
    num_classes,
    classifier_activation="softmax",
    preprocess=None,
    l2_reg=0.01,
):
    """Build MiniXception (see references).

    # Arguments
        input_shape: List of three integers e.g. ``[H, W, 3]``
        num_classes: Int.
        weights: ``None`` or string with pre-trained dataset. Valid datasets
            include only ``FER``.

    # Returns
        Keras model.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    regularization = l2(l2_reg)

    # base
    img_input = Input(input_shape)
    if preprocess is None:
        x = Conv2D(
            5,
            (3, 3),
            strides=(1, 1),
            kernel_regularizer=regularization,
            use_bias=False,
        )(img_input)
    elif preprocess == "rescale":
        x = Rescaling(scale=1 / 127.5, offset=-1)(img_input)
    else:
        raise ValueError("Preprocess must be None 'rescale' or 'normalize'")
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        8,
        (3, 3),
        strides=(1, 1),
        kernel_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # module 1
    residual = Conv2D(
        16, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(
        16,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        16,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(
        32, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(
        32,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        32,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(
        64, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(
        64,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        64,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(
        128, (1, 1), strides=(1, 1), padding="same", use_bias=False
    )(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        depthwise_regularizer=regularization,
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation(classifier_activation, name="predictions")(x)

    model = Model(img_input, output)
    return model


def MiniXceptionFER():
    """Build MiniXception trained in FER (see references).

    # Returns
        Keras model.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    model = MiniXception((48, 48, 1), 7)
    filename = "fer2013_mini_XCEPTION.hdf5"
    URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.6/"
    path = get_file(filename, URL + filename, cache_subdir="paz/models")
    model.load_weights(path)
    return model


def build_mini_xception_imdb(input_shape, num_classes, l2_reg=0.01):
    # Original face_classification mini_XCEPTION: 8-kernel stem and a strided,
    # max-pooled fourth module. Kept faithful so the released IMDB weights load
    # by topological order. Rescaling maps the [0, 1] classifier input to the
    # [-1, 1] training range.
    reg = l2(l2_reg)
    image = Input(input_shape)
    x = Rescaling(scale=2.0, offset=-1.0)(image)
    x = Conv2D(8, 3, kernel_regularizer=reg, use_bias=False)(x)
    x = Activation("relu")(BatchNormalization()(x))
    x = Conv2D(8, 3, kernel_regularizer=reg, use_bias=False)(x)
    x = Activation("relu")(BatchNormalization()(x))
    for num_kernels in [16, 32, 64, 128]:
        residual = Conv2D(num_kernels, 1, strides=2, padding="same",
                          use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(num_kernels, 3, padding="same",
                            depthwise_regularizer=reg, use_bias=False)(x)
        x = Activation("relu")(BatchNormalization()(x))
        x = SeparableConv2D(num_kernels, 3, padding="same",
                            depthwise_regularizer=reg, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, residual])
    x = Conv2D(num_classes, 3, padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation("softmax", name="predictions")(x)
    return Model(image, output)


def MiniXceptionIMDB():
    """Build mini_XCEPTION with the released IMDB two-class face weights.

    # Returns
        Keras model.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    model = build_mini_xception_imdb((64, 64, 1), 2)
    filename = "imdb_mini_XCEPTION_paz_jax.weights.h5"
    URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.18/"
    path = get_file(filename, URL + filename, cache_subdir="paz/models")
    model.load_weights(path)
    return model
