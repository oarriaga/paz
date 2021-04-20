import re
import tensorflow as tf
import efficientnet_model
import utils
import functools

_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25',
    'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25',
    'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25',
    'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]


def efficientnet(
    width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, survival_prob=0.8
):
    """Creates a efficientnet model."""
    global_params = efficientnet_model.GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        survival_prob=survival_prob,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=tf.nn.swish,
        # The default is TPU-specific batch norm.
        # The alternative is tf.layers.BatchNormalization.
        batch_norm=utils.TpuBatchNormalization,  # TPU-specific requirement.
        use_se=True,
        clip_projection_output=False,
    )
    return global_params


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


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
            condconv=('cc' in block_string),
        )

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters,
            'c%d' % block.conv_type,
            'f%d' % block.fused_conv,
            'p%d' % block.super_pixel,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
            args.append('noskip')
        if block.condconv:
            args.append('cc')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.

        # Arguments
          string_list: a list of strings, each string is a notation of block.

        # Returns
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.

        # Arguments
          blocks_args: A list of namedtuples to represent blocks arguments.
        # Returns
          block_strings: a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model."""
    if model_name.startswith('efficientnet'):
        width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_params(
            model_name
        )
        global_params = efficientnet(width_coefficient, depth_coefficient, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    blocks_args = decoder.decode(global_params.blocks_args)

    # print('global_params= %s', global_params)
    return blocks_args, global_params


def build_model_base(model_name, override_params=None):
    """Create a base feature network and return the features before pooling.

    # Arguments
      images: input images tensor.
      model_name: string, the predefined model name.
      training: boolean, whether the model is constructed for training.
      override_params: A dictionary of params for overriding. Fields must exist in
        efficientnet_model.GlobalParams.

    # Returns
      features: base features before pooling.
      endpoints: the endpoints for each layer.

    # Raises
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """
    # For backward compatibility.
    if override_params and override_params.get('drop_connect_rate', None):
        override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

    blocks_args, global_params = get_model_params(model_name, override_params)

    model = efficientnet_model.Model(blocks_args, global_params, model_name)
    return model


def build_backbone(config):
    """
    EfficientDet backbone builder.
    # Arguments
        features: Tensor, indicating the image input to the architecture.
        model_name: String, indicating the EfficientDet-Dx architecture name, x denotes the EfficientDet type.
        backbone_name: String, indicating the EfficientNet-Bx backbone used for feature extraction by EfficientDet-Dx model.
    # Returns
        class_outputs: Tensor, indicating the class probability scores from the classification head.
        box_outputs: Tensor, indicating the box regression outputs from the box regression head.
    """

    # Get config from the hparams
    # TODO: Modify after model completion to argparse and defaults
    is_training_bn = config.is_training_bn
    backbone_name = config.backbone_name
    if 'efficientnet' in backbone_name:
        override_params = {
            'batch_norm': utils.batch_norm_class(is_training_bn, config.strategy),
            'relu_fn': functools.partial(utils.activation_fn, act_type=config.act_type),
        }
        if 'b0' in backbone_name:
            override_params['survival_prob'] = 0.0
        if config.backbone_config is not None:
            override_params['blocks_args'] = BlockDecoder().encode(
                config.backbone_config.blocks
            )
        override_params['data_format'] = config.data_format

        model = build_model_base(backbone_name, override_params)
    else:
        raise ValueError('backbone model {} is not supported.'.format(backbone_name))

    return model

