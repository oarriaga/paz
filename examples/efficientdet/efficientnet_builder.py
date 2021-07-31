from tensorflow.keras.layers import BatchNormalization
import efficientnet_model
import re


_DEFAULT_BLOCKS_ARGS = ['r1_k3_s11_e1_i32_o16_se0.25',
                        'r2_k3_s22_e6_i16_o24_se0.25',
                        'r2_k5_s22_e6_i24_o40_se0.25',
                        'r3_k3_s22_e6_i40_o80_se0.25',
                        'r3_k5_s11_e6_i80_o112_se0.25',
                        'r4_k5_s22_e6_i112_o192_se0.25',
                        'r1_k3_s11_e6_i192_o320_se0.25']


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, survival_rate=0.8):
    """Creates efficientnet model."""
    global_params = efficientnet_model.GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS, batch_norm=BatchNormalization,
        dropout_rate=dropout_rate, survival_rate=survival_rate,
        data_format='channels_last', num_classes=90,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8, min_depth=None, activation='swish',
        use_se=True, clip_projection_output=False)
    return global_params


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
        params: Dictionary, parameters for building the model

    # Returns:
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


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return efficientnet_model.BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])],
            conv_type=int(options['c']) if 'c' in options else 0,
            fused_conv=int(options['f']) if 'f' in options else 0,
            super_pixel=int(options['p']) if 'p' in options else 0,
            condconv=('cc' in block_string)
        )

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks
        inside the network.

        # Arguments
            string_list: a list of strings, each string is a notation of block.

        # Returns
            A list of namedtuples to represent block arguments.
        """
        assert isinstance(string_list, list)
        block_args = []
        for block_string in string_list:
            block_args.append(self._decode_block_string(block_string))
        return block_args


def get_model_params(model_name, params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        efficientnet_param = get_efficientnet_params(model_name)
        width_coefficient, depth_coefficient, dropout_rate = efficientnet_param
        global_params = efficientnet(
            width_coefficient, depth_coefficient, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' %
                                  model_name)
    if params:
        # ValueError will be raised here if params has fields
        # not in global_params
        global_params = global_params._replace(**params)
    decoder = BlockDecoder()
    block_args = decoder.decode(global_params.blocks_args)
    return block_args, global_params


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
    blocks_args, global_params = get_model_params(model_name, params)
    model = efficientnet_model.Model(blocks_args, global_params, model_name)
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
