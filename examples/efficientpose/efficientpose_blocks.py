import tensorflow as tf
from tensorflow.keras.layers import (GroupNormalization, Concatenate,
                                     Add, Reshape)
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_head_conv2D)


def build_pose_estimator_head(middles, subnet_iterations, subnet_repeats,
                              num_anchors, num_filters, num_pose_dims):

    args = (middles, subnet_iterations, subnet_repeats,
            num_anchors, num_filters)
    rotations = RotationNet(*args, num_pose_dims)
    rotations = Concatenate(axis=1, name='rotation')(rotations)
    translations = TranslationNet(*args)
    translations = Concatenate(axis=1, name='translation_raw')(translations)
    return rotations, translations


def RotationNet(middles, subnet_iterations, subnet_repeats,
                num_anchors, num_filters, num_pose_dims):

    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_pose_dims * num_anchors]
    rotations = build_rotation_head(middles, subnet_repeats,
                                    num_filters, bias_initializer)
    return build_iterative_rotation_subnet(
        *rotations, subnet_iterations, subnet_repeats - 1,
        num_filters, bias_initializer, num_pose_dims)


def build_rotation_head(features, subnet_repeats, num_filters,
                        bias_initializer, gn_groups=4, gn_axis=-1):

    conv_blocks = build_head_conv2D(subnet_repeats, num_filters[0],
                                    bias_initializer)
    head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    rotation_features, initial_rotations = [], []
    for x in features:
        for block_arg in range(subnet_repeats):
            x = conv_blocks[block_arg](x)
            x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
            x = tf.nn.swish(x)
        initial_rotation = head_conv(x)
        rotation_features.append(x)
        initial_rotations.append(initial_rotation)
    return rotation_features, initial_rotations


def build_iterative_rotation_subnet(rotation_features, initial_rotations,
                                    subnet_iterations, subnet_repeats,
                                    num_filters, bias_initializer,
                                    num_pose_dims, gn_groups=4, gn_axis=-1):

    conv_blocks = build_head_conv2D(subnet_repeats, num_filters[0],
                                    bias_initializer)
    head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    rotations = []
    for x, initial_rotation in zip(rotation_features, initial_rotations):
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([x, initial_rotation])
            for block_arg in range(subnet_repeats):
                x = conv_blocks[block_arg](x)
                x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
                x = tf.nn.swish(x)
            delta_rotation = head_conv(x)
            initial_rotation = Add()([initial_rotation, delta_rotation])
        rotation = Reshape((-1, num_pose_dims))(initial_rotation)
        rotations.append(rotation)
    return rotations


def TranslationNet(middles, subnet_iterations, subnet_repeats,
                   num_anchors, num_filters):

    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_anchors * 2, num_anchors]
    translations = build_translation_head(middles, subnet_repeats,
                                          num_filters, bias_initializer)
    return build_iterative_translation_subnet(
        *translations, subnet_repeats - 1, num_filters,
        subnet_iterations, bias_initializer)


def build_translation_head(features, subnet_repeats, num_filters,
                           bias_initializer, gn_groups=4, gn_axis=-1):

    conv_blocks = build_head_conv2D(subnet_repeats, num_filters[0],
                                    bias_initializer)
    head_xy_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    head_z_conv = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    translation_features, translations_xy, translations_z = [], [], []
    for x in features:
        for block_arg in range(subnet_repeats):
            x = conv_blocks[block_arg](x)
            x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
            x = tf.nn.swish(x)
        translation_xy = head_xy_conv(x)
        translation_z = head_z_conv(x)
        translation_features.append(x)
        translations_xy.append(translation_xy)
        translations_z.append(translation_z)
    return translation_features, translations_xy, translations_z


def build_iterative_translation_subnet(translation_features, translations_xy,
                                       translations_z, subnet_repeats,
                                       num_filters, subnet_iterations,
                                       bias_initializer, gn_groups=4,
                                       gn_axis=-1):

    conv_blocks = build_head_conv2D(subnet_repeats, num_filters[0],
                                    bias_initializer)
    head_xy = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    head_z = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    translations = []
    iterator = zip(translation_features, translations_xy, translations_z)
    for x, translation_xy, translation_z in iterator:
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([x, translation_xy, translation_z])
            for block_arg in range(subnet_repeats):
                x = conv_blocks[block_arg](x)
                x = GroupNormalization(groups=gn_groups, axis=gn_axis)(x)
                x = tf.nn.swish(x)
            delta_translation_xy = head_xy(x)
            delta_translation_z = head_z(x)
            translation_xy = Add()([translation_xy, delta_translation_xy])
            translation_z = Add()([translation_z, delta_translation_z])

        translation_xy = Reshape((-1, 2))(translation_xy)
        translation_z = Reshape((-1, 1))(translation_z)
        translation = Concatenate(axis=-1)([translation_xy, translation_z])
        translations.append(translation)
    return translations
