from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from ..layers import ExpectedValue2D


def dense_block(x, blocks, growth_rate):
    for block_arg in range(blocks):
        x1 = Conv2D(4 * growth_rate, 1, use_bias=False)(x)
        x1 = BatchNormalization(epsilon=1.001e-5)(x)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False)(x1)
        x1 = BatchNormalization(epsilon=1.001e-5)(x1)
        x1 = Activation('relu')(x1)
        x = Concatenate(axis=-1)([x, x1])
    return x


def residual_block(x, num_kernels, strides=1):
    residual = x
    x = Conv2D(num_kernels, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_kernels, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x


def transition_block(x, alpha):
    filters = int(K.int_shape(x)[-1] * alpha)
    x = Conv2D(filters, 1, strides=2, use_bias=False)(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    return x


def stem(x, filters):
    x = Conv2D(filters, 3, padding='same', strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization(epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same', strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization(epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    return x


def fuse(tensors, base_kernels=32):
    all_tensors = []
    for x_tensor_arg, x in enumerate(tensors):
        x_to_y_tensors = []
        for y_tensor_arg in range(len(tensors)):
            # step: how much the feature map is upsampled or downsampled
            steps = x_tensor_arg - y_tensor_arg

            if steps == 0:
                num_kernels = K.int_shape(x)[-1]
                y = Conv2D(num_kernels, 3, padding='same',
                           strides=1, use_bias=False)(x)
                y = BatchNormalization(epsilon=1.1e-5)(y)
                y = Activation('relu')(y)

            if steps < 0:
                y = x
                for step in range(abs(steps)):
                    num_kernels = int(K.int_shape(x)[-1] * (step + 1))
                    y = Conv2D(num_kernels, 3, strides=2,
                               padding='same', use_bias=False)(y)
                    y = BatchNormalization(epsilon=1.1e-5)(y)
                    y = Activation('relu')(y)

            if steps > 0:
                num_kernels = int(K.int_shape(x)[-1] / steps)
                y = Conv2D(num_kernels, 1, use_bias=False)(x)
                y = BatchNormalization(epsilon=1.1e-5)(y)
                y = Activation('relu')(y)
                y = UpSampling2D(size=(2**steps, 2**steps))(y)

            x_to_y_tensors.append(y)
        all_tensors.append(x_to_y_tensors)

    output_tensors = []
    for reciever_arg in range(len(tensors)):
        same_resolution_tensors = []
        for giver_arg in range(len(tensors)):
            tensor = all_tensors[giver_arg][reciever_arg]
            same_resolution_tensors.append(tensor)
        x = Concatenate()(same_resolution_tensors)
        num_kernels = base_kernels * (2 ** (reciever_arg))
        x = Conv2D(num_kernels, 1, use_bias=False)(x)
        x = BatchNormalization(epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        output_tensors.append(x)
    return output_tensors


def bottleneck(x, filters=64, expansion=4):
    residual = x
    x = Conv2D(filters, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters * expansion, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)
    return x


def HRNetDense(input_shape=(128, 128, 3), num_keypoints=20, growth_rate=4):
    # stem
    inputs = Input(shape=input_shape)
    x1 = stem(inputs, 64)
    x1 = Conv2D(64 * 4, 1, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    for block in range(4):
        x1 = bottleneck(x1)

    # stage I
    x1 = Conv2D(32, 3, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x2 = transition_block(x1, 2)
    print('stage 1', x1.shape, x2.shape)

    # stage II
    x1 = dense_block(x1, 4, growth_rate)
    x2 = dense_block(x2, 4, growth_rate)
    x1, x2 = fuse([x1, x2])
    x3 = transition_block(x2, 0.5)
    print('stage 2', x1.shape, x2.shape, x3.shape)

    # stage III
    x1 = dense_block(x1, 4, growth_rate)
    x2 = dense_block(x2, 4, growth_rate)
    x3 = dense_block(x3, 4, growth_rate)
    x1, x2, x3 = fuse([x1, x2, x3])
    x4 = transition_block(x3, 0.5)
    print('stage 3', x1.shape, x2.shape, x3.shape, x4.shape)

    # stage IV
    x1 = dense_block(x1, 3, growth_rate)
    x2 = dense_block(x2, 3, growth_rate)
    x3 = dense_block(x3, 3, growth_rate)
    x4 = dense_block(x4, 3, growth_rate)
    x1, x2, x3, x4 = fuse([x1, x2, x3, x4])
    print('stage 4', x1.shape, x2.shape, x3.shape, x4.shape)

    x2 = UpSampling2D(size=(2, 2))(x2)
    x3 = UpSampling2D(size=(4, 4))(x3)
    x4 = UpSampling2D(size=(8, 8))(x4)
    x = Concatenate()([x1, x2, x3, x4])

    # head
    x = Conv2D(480, 1)(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_keypoints, 1)(x)

    # extra
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = Permute([3, 1, 2])(x)
    x = Reshape([num_keypoints, input_shape[0] * input_shape[1]])(x)
    x = Activation('softmax')(x)
    x = Reshape([num_keypoints, input_shape[0], input_shape[1]])(x)
    outputs = ExpectedValue2D(name='expected_uv')(x)
    model = Model(inputs, outputs, name='hrnet-dense')
    return model


def HRNetResidual(input_shape=(128, 128, 3), num_keypoints=20):
    """Instantiates HRNET Residual model

    # Arguments
        input_shape: List of three elements e.g. ''(H, W, 3)''
        num_keypoints: Int.

    # Returns
        Tensorflow-Keras model.

    # References
       -[High-Resolution Representations for Labeling Pixels
            and Regions](https://arxiv.org/pdf/1904.04514.pdf)
    """

    # stem
    inputs = Input(shape=input_shape, name='image')
    x1 = stem(inputs, 64)
    x1 = Conv2D(64 * 4, 1, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    for block in range(4):
        x1 = bottleneck(x1)

    # stage I
    x1 = Conv2D(32, 3, padding='same', use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x2 = transition_block(x1, 2)

    # stage II
    for block in range(4):
        x1 = residual_block(x1, 32)
        x2 = residual_block(x2, 64)
    x1, x2 = fuse([x1, x2])
    x3 = transition_block(x2, 2)

    # stage III
    for module in range(4):
        for block in range(4):
            x1 = residual_block(x1, 32)
            x2 = residual_block(x2, 64)
            x3 = residual_block(x3, 128)
        x1, x2, x3 = fuse([x1, x2, x3])
    x4 = transition_block(x3, 2)

    # stage IV
    for module in range(3):
        for block in range(4):
            x1 = residual_block(x1, 32)
            x2 = residual_block(x2, 64)
            x3 = residual_block(x3, 128)
            x4 = residual_block(x4, 256)
        x1, x2, x3, x4 = fuse([x1, x2, x3, x4])

    # head
    x2 = UpSampling2D(size=(2, 2))(x2)
    x3 = UpSampling2D(size=(4, 4))(x3)
    x4 = UpSampling2D(size=(8, 8))(x4)
    x = Concatenate()([x1, x2, x3, x4])
    x = Conv2D(480, 1)(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(num_keypoints, 1)(x)

    # extra
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = Permute([3, 1, 2])(x)
    x = Reshape([num_keypoints, input_shape[0] * input_shape[1]])(x)
    x = Activation('softmax')(x)
    x = Reshape([num_keypoints, input_shape[0], input_shape[1]])(x)
    outputs = ExpectedValue2D(name='keypoints')(x)
    model = Model(inputs, outputs, name='hrnet-residual')
    return model
