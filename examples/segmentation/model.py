from tensorflow.keras.layers import Conv2DTranspose, Concatenate, UpSampling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16


def convolution_block(inputs, filters, kernel_size=3, activation='relu'):
    kwargs = {'use_bias': False, 'kernel_initializer': 'he_uniform'}
    x = Conv2D(filters, kernel_size, (1, 1), 'same', **kwargs)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def upsample_block(x, filters, branch):
    x = UpSampling2D(size=2)(x)
    x = Concatenate(axis=3)([x, branch])
    x = convolution_block(x, filters)
    x = convolution_block(x, filters)
    return x


def transpose_block(x, filters, branch):
    x = Conv2DTranspose(filters, 4, (2, 2), 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate(axis=3)([x, branch])
    x = convolution_block(x, filters)
    return x


def build_backbone(BACKBONE, shape, branch_names, weights, frozen=False):
    kwargs = {'include_top': False, 'input_shape': shape, 'weights': weights}
    backbone = BACKBONE(**kwargs)
    if frozen:
        for layer in backbone.layers:
            layer.trainable = False
    branch_tensors = []
    for layer_name in branch_names:
        branch_tensors.append(backbone.get_layer(layer_name).output)
    return backbone, branch_tensors


def build_UNET(num_classes, backbone, branch_tensors,
               decoder, decoder_filters, activation, name):
    inputs, x = backbone.input, backbone.output
    if isinstance(backbone.layers[-1], MaxPooling2D):
        x = convolution_block(x, 512)
        x = convolution_block(x, 512)

    for branch, filters in zip(branch_tensors, decoder_filters):
        x = decoder(x, filters, branch)

    kwargs = {'use_bias': True, 'kernel_initializer': 'glorot_uniform'}
    x = Conv2D(num_classes, 3, (1, 1), 'same', **kwargs)(x)
    outputs = Activation(activation, name='masks')(x)
    model = Model(inputs, outputs, name=name)
    return model


def UNET(input_shape, num_classes, branch_names, BACKBONE, weights,
         freeze_backbone=False, activation='sigmoid', decoder_type='upsample',
         decoder_filters=[256, 128, 64, 32, 16], name='UNET'):

    args = [BACKBONE, input_shape, branch_names, weights, freeze_backbone]
    backbone, branch_tensors = build_backbone(*args)
    if decoder_type == 'upsample':
        decoder = upsample_block
    if decoder_type == 'transpose':
        decoder = transpose_block

    model = build_UNET(num_classes, backbone, branch_tensors, decoder,
                       decoder_filters, activation, name)
    return model


def UNET_VGG16(num_classes=1, input_shape=(224, 224, 3), weights='imagenet',
               freeze=False, activation='sigmoid', decoder_type='upsample',
               decode_filters=[256, 128, 64, 32, 16]):
    VGG16_branches = ['block5_conv3', 'block4_conv3', 'block3_conv3',
                      'block2_conv2', 'block1_conv2']
    return UNET(input_shape, num_classes, VGG16_branches, VGG16, weights,
                freeze, activation, decoder_type, decode_filters, 'UNET_VGG16')


if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    model = UNET_VGG16()
    model.summary()
    plot_model(model, 'unet.png', True, True, dpi=200)
