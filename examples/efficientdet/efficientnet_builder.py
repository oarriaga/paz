from tensorflow.keras.layers import BatchNormalization
import efficientnet_model


def get_efficientnet_params(model_name):
    """Default efficientnet scaling coefficients and
    image name based on model name.
    The value of each model name in the key represents:
    (width_coefficient, depth_coefficient, dropout_rate).
    with_coefficient: scaling coefficient for network width.
    depth_coefficient: scaling coefficient for network depth.
    dropout_rate: dropout rate for final fully connected layers.

    # Arguments
        model_name: String, name of the EfficientNet backbone

    # Returns
        efficientnetparams: Dictionary, parameters corresponding to
        width coefficient, depth coefficient, dropout rate
    """
    efficientnet_params = {'efficientnet-b0': (1.0, 1.0, 0.2),
                           'efficientnet-b1': (1.0, 1.1, 0.2),
                           'efficientnet-b2': (1.1, 1.2, 0.3),
                           'efficientnet-b3': (1.2, 1.4, 0.3),
                           'efficientnet-b4': (1.4, 1.8, 0.4),
                           'efficientnet-b5': (1.6, 2.2, 0.4),
                           'efficientnet-b6': (1.8, 2.6, 0.5),
                           'efficientnet-b7': (2.0, 3.1, 0.5),
                           'efficientnet-b8': (2.2, 3.6, 0.5),
                           'efficientnet-l2': (4.3, 5.3, 0.5)}
    return efficientnet_params[model_name]


def build_model_base(model_name, params=None):
    """Create a base feature network and return the features before pooling.

    # Arguments
        model_name: String, name of the EfficientNet backbone
        params: Dictionary, parameters for building the model

    # Returns:
        model: EfficientNet model

    # Raises
        When model_name specified an undefined model,
        raises NotImplementedError.
        When params has invalid fields, raises ValueError.
    """
    if params and params.get('drop_connect_rate', None):
        params['survival_rate'] = 1 - params['drop_connect_rate']
    efficientnet_param = get_efficientnet_params(model_name)
    width_coefficient, depth_coefficient, dropout_rate = efficientnet_param
    data_format = 'channels_last'
    num_classes = 90
    depth_divisor = 8
    min_depth = None
    survival_rate = 1 - dropout_rate
    activation = 'swish'
    batch_norm = BatchNormalization
    use_squeeze_excitation = True
    local_pooling = None
    clip_projection_output = False
    fix_head_stem = None
    kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
    num_repeats = [1, 2, 2, 3, 3, 4, 1]
    input_filters = [32, 16, 24, 40, 80, 112, 192]
    output_filters = [16, 24, 40, 80, 112, 192, 320]
    expand_ratios = [1, 6, 6, 6, 6, 6, 6]
    strides = [[1, 1], [2, 2], [2, 2], [2, 2],
               [1, 1], [2, 2], [1, 1]]
    squeeze_excite_ratio = 0.25
    use_skip_connection = True
    conv_type = 0
    fused_conv = 0
    super_pixel = 0
    num_blocks = 7
    model = efficientnet_model.EfficientNet(
        dropout_rate, data_format, num_classes, width_coefficient,
        depth_coefficient, depth_divisor, min_depth, survival_rate,
        activation, batch_norm, use_squeeze_excitation,
        local_pooling, clip_projection_output,
        fix_head_stem, kernel_sizes, num_repeats, input_filters,
        output_filters, expand_ratios, strides, squeeze_excite_ratio,
        use_skip_connection, conv_type, fused_conv, super_pixel,
        num_blocks, model_name)
    return model


def build_backbone(backbone_name, activation, survival_rate):
    """
    Build backbone model.

    # Arguments
        config: Configuration of the EfficientDet model.

    # Returns
        EfficientNet model with intermediate feature levels.
    """
    if 'efficientnet' in backbone_name:
        params = {'batch_norm': BatchNormalization, 'activation': activation}
        if 'b0' in backbone_name:
            params['survival_rate'] = 0.0
        else:
            params['survival_rate'] = survival_rate
        model = build_model_base(backbone_name, params)
    else:
        raise ValueError('backbone model {} is not supported.'.format(
            backbone_name))
    return model
