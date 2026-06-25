from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import GroupNormalization
from keras.layers import Reshape
from keras.activations import swish

from ...detection.efficientdet.efficientdet_blocks import build_head
from ...detection.efficientdet.efficientdet_blocks import build_head_conv2D


def build_pose_estimator_head(middles, subnet_iterations, subnet_repeats,
                              num_anchors, num_filters, num_dims):
    args = (middles, subnet_iterations, subnet_repeats, num_anchors)
    rotations = RotationNet(*args, num_filters, num_dims)
    rotations = Concatenate(axis=1)(rotations)
    translations = TranslationNet(*args, num_filters)
    translations = Concatenate(axis=1)(translations)
    merge = Concatenate(axis=-1, name="transformation")
    return merge([rotations, translations])


def RotationNet(middles, subnet_iterations, subnet_repeats, num_anchors,
                num_filters, num_dims, survival_rate=None):
    bias_initializer = "zeros"
    num_filters = [num_filters, num_dims * num_anchors]
    args = (subnet_repeats, num_filters, bias_initializer)
    features, rotations = build_head(middles, *args, survival_rate,
                                     normalization="group")
    return refine_rotation_iteratively(features, rotations, subnet_iterations,
                                       *args, num_dims)


def refine_rotation_iteratively(rotation_features, initial_rotations,
                                subnet_iterations, subnet_repeats,
                                num_filters, bias_initializer, num_dims):
    rotations = []
    args = (subnet_repeats, num_filters, bias_initializer)
    for feature, rotation in zip(rotation_features, initial_rotations):
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([feature, rotation])
            rotation = Add()([rotation, refine_rotation(x, *args)])
        rotations.append(Reshape((-1, num_dims))(rotation))
    return rotations


def refine_rotation(x, repeats, num_filters, bias_initializer,
                    channels_per_group=16):
    conv_body = build_head_conv2D(repeats, num_filters[0], bias_initializer)
    conv_head = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    num_groups = int(num_filters[0] / channels_per_group)
    for block_arg in range(repeats):
        x = conv_body[block_arg](x)
        x = GroupNormalization(groups=num_groups)(x)
        x = swish(x)
    return conv_head(x)


def TranslationNet(middles, subnet_iterations, subnet_repeats, num_anchors,
                   num_filters):
    bias_initializer = "zeros"
    num_filters = [num_filters, num_anchors * 2, num_anchors]
    args = (subnet_repeats, num_filters, bias_initializer)
    regressions = regress_initial_translations(middles, *args)
    return refine_translation_iteratively(*regressions, *args,
                                          subnet_iterations)


def regress_initial_translations(middles, subnet_repeats, num_filters,
                                 bias_initializer):
    features, translations_xy, translations_z = [], [], []
    args = (subnet_repeats, num_filters, bias_initializer)
    for x in middles:
        x, translation_xy, translation_z = build_translation_subnets(x, *args)
        features.append(x)
        translations_xy.append(translation_xy)
        translations_z.append(translation_z)
    return [features, translations_xy, translations_z]


def build_translation_subnets(x, repeats, num_filters, bias_initializer,
                              channels_per_group=16):
    conv_body = build_head_conv2D(repeats, num_filters[0], bias_initializer)
    conv_head_xy = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    conv_head_z = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    num_groups = int(num_filters[0] / channels_per_group)
    for block_arg in range(repeats):
        x = conv_body[block_arg](x)
        x = GroupNormalization(groups=num_groups)(x)
        x = swish(x)
    return [x, conv_head_xy(x), conv_head_z(x)]


def refine_translation_iteratively(translation_features, translations_xy,
                                   translations_z, subnet_repeats, num_filters,
                                   bias_initializer, subnet_iterations):
    translations = []
    args = (subnet_repeats, num_filters, bias_initializer)
    iterator = zip(translation_features, translations_xy, translations_z)
    for feature, translation_xy, translation_z in iterator:
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([feature, translation_xy, translation_z])
            delta_xy, delta_z = refine_translation(x, *args)
            translation_xy = Add()([translation_xy, delta_xy])
            translation_z = Add()([translation_z, delta_z])
        translation_xy = Reshape((-1, 2))(translation_xy)
        translation_z = Reshape((-1, 1))(translation_z)
        translation = Concatenate(axis=-1)([translation_xy, translation_z])
        translations.append(translation)
    return translations


def refine_translation(x, repeats, num_filters, bias_initializer,
                       channels_per_group=16):
    conv_body = build_head_conv2D(repeats, num_filters[0], bias_initializer)
    conv_head_xy = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    conv_head_z = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    num_groups = int(num_filters[0] / channels_per_group)
    for block_arg in range(repeats):
        x = conv_body[block_arg](x)
        x = GroupNormalization(groups=num_groups)(x)
        x = swish(x)
    return [conv_head_xy(x), conv_head_z(x)]
