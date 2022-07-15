from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Add, Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file


URL = 'https://github.com/oarriaga/altamira-data/releases/download/v0.6/'


def xception_block(input_tensor, num_kernels, l2_reg=0.01):
    """Xception core block.

    # Arguments
        input_tenso: Keras tensor.
        num_kernels: Int. Number of convolutional kernels in block.
        l2_reg: Float. l2 regression.

    # Returns
        output tensor for the block.
    """
    residual = Conv2D(num_kernels, 1, strides=(2, 2),
                      padding='same', use_bias=False)(input_tensor)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(
        num_kernels, 3, padding='same',
        kernel_regularizer=l2(l2_reg), use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(num_kernels, 3, padding='same',
                        kernel_regularizer=l2(l2_reg), use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    return x


def build_xception(
        input_shape, num_classes, stem_kernels, block_kernels, l2_reg=0.01):
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

    x = inputs = Input(input_shape, name='image')
    for num_kernels in stem_kernels:
        x = Conv2D(num_kernels, 3, kernel_regularizer=l2(l2_reg),
                   use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for num_kernels in block_kernels:
        x = xception_block(x, num_kernels, l2_reg)

    x = Conv2D(num_classes, 3, kernel_regularizer=l2(l2_reg),
               padding='same')(x)
    # x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='label')(x)

    model_name = '-'.join(['XCEPTION',
                           str(input_shape[0]),
                           str(stem_kernels[0]),
                           str(len(block_kernels))
                           ])
    model = Model(inputs, output, name=model_name)
    return model


def MiniXception(input_shape, num_classes, weights=None):
    """Build MiniXception (see references).

    # Arguments
        input_shape: List of three integers e.g. ``[H, W, 3]``
        num_classes: Int.
        weights: ``None`` or string with pre-trained dataset. Valid datasets
            include only ``FER``.

    # Returns
        Tensorflow-Keras model.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    if weights == 'FER':
        filename = 'fer2013_mini_XCEPTION.119-0.65.hdf5'
        path = get_file(filename, URL + filename, cache_subdir='paz/models')
        model = load_model(path)
    else:
        stem_kernels = [32, 64]
        block_data = [128, 128, 256, 256, 512, 512, 1024]
        model_inputs = (input_shape, num_classes, stem_kernels, block_data)
        model = build_xception(*model_inputs)
    model._name = 'MINI-XCEPTION'
    return model
