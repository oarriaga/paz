from tensorflow.keras.layers import Conv2DTranspose, Concatenate, UpSampling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, Input
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50V2


def convolution_block(inputs, filters, kernel_size=3, activation='relu'):
    """UNET convolution block containing Conv2D -> BatchNorm -> Activation

    # Arguments
        inputs: Keras/tensorflow tensor input.
        filters: Int. Number of filters.
        kernel_size: Int. Kernel size of convolutions.
        activation: String. Activation used convolution.

    # Returns
        Keras/tensorflow tensor.
    """
    kwargs = {'use_bias': False, 'kernel_initializer': 'he_uniform'}
    x = Conv2D(filters, kernel_size, (1, 1), 'same', **kwargs)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def upsample_block(x, filters, branch):
    """UNET upsample block. This block upsamples ``x``, concatenates a
    ``branch`` tensor and applies two convolution blocks:
    Upsample -> Concatenate -> 2 x ConvBlock.

    # Arguments
        x: Keras/tensorflow tensor.
        filters: Int. Number of filters
        branch: Tensor to be concatated to the upsamples ``x`` tensor.

    # Returns
        A Keras tensor.
    """
    x = UpSampling2D(size=2)(x)
    x = Concatenate(axis=3)([x, branch])
    x = convolution_block(x, filters)
    x = convolution_block(x, filters)
    return x


def transpose_block(x, filters, branch):
    """UNET transpose block. This block upsamples ``x``, concatenates a
    ``branch`` tensor and applies two convolution blocks:
    Conv2DTranspose -> Concatenate -> 2 x ConvBlock.

    # Arguments
        x: Keras/tensorflow tensor.
        filters: Int. Number of filters
        branch: Tensor to be concatated to the upsamples ``x`` tensor.

    # Returns
        A Keras tensor.
    """
    x = Conv2DTranspose(filters, 4, (2, 2), 'same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate(axis=3)([x, branch])
    x = convolution_block(x, filters)
    return x


def freeze_model(model):
    """Freezes gradient pass for the entire model

    # Arguments:
        model: Keras/tensorflow model

    # Returns:
        A Keras/tensorflow model
    """
    for layer in model.layers:
        layer.trainable = False
    return model


def get_tensors(model, layer_names):
    """Gets all the tensor outputs of the given layer names.

    # Arguments
        model: Keras/tensorflow model.
        layer_names: List of strings which each string is a layer name.

    # Returns
        List of Keras tensors.
    """
    tensors = []
    for layer_name in layer_names:
        tensors.append(model.get_layer(layer_name).output)
    return model, tensors


def build_backbone(BACKBONE, shape, branch_names, weights,
                   frozen=False, input_tensor=None):
    """Builds ``BACKBONE`` class for UNET model.

    # Arguments
        BACKBONE: Class for instantiating a backbone model
        shape: List of integers: ``(H, W, num_channels)``.
        branch_names: List of strings containing layer names of ``BACKBONE()``.
        weights: String or ``None``.
        frozen: Boolean. If True ``BACKBONE()`` updates are frozen.
        input_tensor: Input tensor. If given ``shape`` is overwritten and this
            tensor is used instead as input.

    # Returns
    """
    kwargs = {'include_top': False, 'input_shape': shape, 'weights': weights}
    if input_tensor is not None:
        kwargs.pop('input_shape')
        kwargs['input_tensor'] = input_tensor
    backbone = BACKBONE(**kwargs)

    if frozen:
        backbone = freeze_model(backbone)

    backbone, branch_tensors = get_tensors(backbone, branch_names)
    return backbone, branch_tensors


def build_UNET(num_classes, backbone, branch_tensors,
               decoder, decoder_filters, activation, name):
    """Build UNET with a given ``backbone`` model.

    # Arguments
        num_classes: Integer used for output number of channels.
        backbone: Instantiated backbone model.
        branch_tensors: List of tensors from ``backbone`` model
        decoder: Function used for upsampling and decoding the output.
        decoder_filters: List of integers used in each application of decoder.
        activation: Output activation of the model.
        name: String. indicating the name of the model.

    # Returns
        A UNET Keras/tensorflow model.
    """
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
         decoder_filters=[256, 128, 64, 32, 16], input_tensor=None,
         name='UNET'):
    """Build a generic UNET model with a given ``BACKBONE`` class.

    # Arguments
        input_shape: List of integers: ``(H, W, num_channels)``.
        num_classes: Integer used for output number of channels.
        branch_names: List of strings containing layer names of ``BACKBONE()``.
        BACKBONE: Class for instantiating a backbone model
        weights: String indicating backbone weights e.g.
            ''imagenet'', ``None``.
        freeze_backbone: Boolean. If True ``BACKBONE()`` updates are frozen.
        decoder_type: String indicating decoding function e.g.
            ''upsample ''transpose''.
        decoder_filters: List of integers used in each application of decoder.
        activation: Output activation of the model.
        input_tensor: Input tensor. If given ``shape`` is overwritten and this
            tensor is used instead as input.
        name: String. indicating the name of the model.

    # Returns
        A UNET Keras/tensorflow model.
    """
    args = [BACKBONE, input_shape, branch_names,
            weights, freeze_backbone, input_tensor]
    backbone, branch_tensors = build_backbone(*args)
    if decoder_type == 'upsample':
        decoder = upsample_block
    if decoder_type == 'transpose':
        decoder = transpose_block

    model = build_UNET(num_classes, backbone, branch_tensors, decoder,
                       decoder_filters, activation, name)
    return model


def UNET_VGG16(num_classes=1, input_shape=(224, 224, 3), weights='imagenet',
               freeze_backbone=False, activation='sigmoid',
               decoder_type='upsample',
               decode_filters=[256, 128, 64, 32, 16]):
    """Build a UNET model with a ``VGG16`` backbone.

    # Arguments
        input_shape: List of integers: ``(H, W, num_channels)``.
        num_classes: Integer used for output number of channels.
        branch_names: List of strings containing layer names of ``BACKBONE()``.
        BACKBONE: Class for instantiating a backbone model
        weights: String indicating backbone weights e.g.
            ''imagenet'', ``None``.
        freeze_backbone: Boolean. If True ``BACKBONE()`` updates are frozen.
        decoder_type: String indicating decoding function e.g.
            ''upsample ''transpose''.
        decoder_filters: List of integers used in each application of decoder.
        activation: Output activation of the model.
        input_tensor: Input tensor. If given ``shape`` is overwritten and this
            tensor is used instead as input.
        name: String. indicating the name of the model.

    # Returns
        A UNET-VGG16 Keras/tensorflow model.
    """
    VGG16_branches = ['block5_conv3', 'block4_conv3', 'block3_conv3',
                      'block2_conv2', 'block1_conv2']
    return UNET(input_shape, num_classes, VGG16_branches, VGG16, weights,
                freeze_backbone, activation, decoder_type, decode_filters,
                name='UNET-VGG16')


def UNET_VGG19(num_classes=1, input_shape=(224, 224, 3), weights='imagenet',
               freeze_backbone=False, activation='sigmoid',
               decoder_type='upsample',
               decode_filters=[256, 128, 64, 32, 16]):
    """Build a UNET model with a ``VGG19`` backbone.

    # Arguments
        input_shape: List of integers: ``(H, W, num_channels)``.
        num_classes: Integer used for output number of channels.
        branch_names: List of strings containing layer names of ``BACKBONE()``.
        BACKBONE: Class for instantiating a backbone model
        weights: String indicating backbone weights e.g.
            ''imagenet'', ``None``.
        freeze_backbone: Boolean. If True ``BACKBONE()`` updates are frozen.
        decoder_type: String indicating decoding function e.g.
            ''upsample ''transpose''.
        decoder_filters: List of integers used in each application of decoder.
        activation: Output activation of the model.
        input_tensor: Input tensor. If given ``shape`` is overwritten and this
            tensor is used instead as input.
        name: String. indicating the name of the model.

    # Returns
        A UNET-VGG19 Keras/tensorflow model.
    """

    VGG19_branches = ['block5_conv4', 'block4_conv4', 'block3_conv4',
                      'block2_conv2', 'block1_conv2']
    return UNET(input_shape, num_classes, VGG19_branches, VGG19, weights,
                freeze_backbone, activation, decoder_type, decode_filters,
                name='UNET-VGG19')


def UNET_RESNET50(num_classes=1, input_shape=(224, 224, 3), weights='imagenet',
                  freeze_backbone=False, activation='sigmoid',
                  decoder_type='upsample',
                  decode_filters=[256, 128, 64, 32, 16]):
    """Build a UNET model with a ``RESNET50V2`` backbone.

    # Arguments
        input_shape: List of integers: ``(H, W, num_channels)``.
        num_classes: Integer used for output number of channels.
        branch_names: List of strings containing layer names of ``BACKBONE()``.
        BACKBONE: Class for instantiating a backbone model
        weights: String indicating backbone weights e.g.
            ''imagenet'', ``None``.
        freeze_backbone: Boolean. If True ``BACKBONE()`` updates are frozen.
        decoder_type: String indicating decoding function e.g.
            ''upsample ''transpose''.
        decoder_filters: List of integers used in each application of decoder.
        activation: Output activation of the model.
        input_tensor: Input tensor. If given ``shape`` is overwritten and this
            tensor is used instead as input.
        name: String. indicating the name of the model.

    # Returns
        A UNET-RESNET50V2 Keras/tensorflow model.
    """
    RESNET50_branches = ['conv4_block6_1_relu', 'conv3_block4_1_relu',
                         'conv2_block3_1_relu', 'conv1_conv', 'input_resnet50']
    input_tensor = Input(input_shape, name='input_resnet50')
    return UNET(input_shape, num_classes, RESNET50_branches, ResNet50V2,
                weights, freeze_backbone, activation, decoder_type,
                decode_filters, input_tensor, 'UNET-RESNET50')
