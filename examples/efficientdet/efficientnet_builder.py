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


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments.

        # Arguments
            block_string: String, denoting the efficientnet block parameters.
        """
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
        block_args = efficientnet_model.BlockArgs
        block_args["num_repeat"] = int(options['r'])
        block_args["kernel_size"] = int(options['k'])
        block_args["strides"] = [int(options['s'][0]), int(options['s'][1])]
        block_args["expand_ratio"] = int(options['e'])
        block_args["input_filters"] = int(options['i'])
        block_args["output_filters"] = int(options['o'])
        if 'se' in options:
            block_args["squeeze_excite_ratio"] = float(options['se'])
        else:
            block_args["squeeze_excite_ratio"] = None
        block_args["id_skip"] = ('noskip' not in block_string)
        block_args["conv_type"] = int(options['c']) if 'c' in options else 0
        block_args["fused_conv"] = int(options['f']) if 'f' in options else 0
        block_args["super_pixel"] = int(options['p']) if 'p' in options else 0
        block_args["condconv"] = ('cc' in block_string)
        return block_args

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
            block_args.append((self._decode_block_string(block_string)).copy())
        return block_args


def get_blocks_args():
    blocks_args_string = _DEFAULT_BLOCKS_ARGS
    decoder = BlockDecoder()
    blocks_args = decoder.decode(blocks_args_string)
    return blocks_args


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
    condconv_num_experts = None
    clip_projection_output = False
    fix_head_stem = None
    blocks_args = get_blocks_args()
    model = efficientnet_model.EfficientNet(
        blocks_args, dropout_rate, data_format, num_classes,
        width_coefficient, depth_coefficient, depth_divisor, min_depth,
        survival_rate, activation, batch_norm, use_squeeze_excitation,
        local_pooling, condconv_num_experts, clip_projection_output,
        fix_head_stem, model_name)
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
