import tensorflow as tf
from tensorflow.keras.layers import (GroupNormalization, Concatenate,
                                     Add, Reshape)
from ...detection.efficientdet.efficientdet_blocks import (
    build_head_conv2D, build_head)


def build_pose_estimator_head(middles, subnet_iterations, subnet_repeats,
                              num_anchors, num_filters, num_dims):
    """Builds EfficientPose pose estimator head
    containing RotationNet and TranslationNet for
    estimation of rotation and translation of the object respectively.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_anchors: List, number of combinations of anchor box's scale
            and aspect ratios.
        num_filters: Int, number of subnet filters.
        num_dims: Int, number of pose dimensions.

    # Returns
        Tensor: Concatenation of estimated rotations and translations
            of shape `(None, num_boxes, num_dims + num_dims)`
    """
    args = (middles, subnet_iterations, subnet_repeats, num_anchors)
    rotations = RotationNet(*args, num_filters, num_dims)
    rotations = Concatenate(axis=1)(rotations)
    translations = TranslationNet(*args, num_filters)
    translations = Concatenate(axis=1)(translations)
    concatenate_transformation = Concatenate(axis=-1, name='transformation')
    return concatenate_transformation([rotations, translations])


def RotationNet(middles, subnet_iterations, subnet_repeats, num_anchors,
                num_filters, num_dims, survival_rate=None):
    """Initializes RotationNet.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_anchors: List, number of combinations of
            anchor box's scale and aspect ratios.
        num_filters: Int, number of subnet filters.
        num_dims: Int, number of pose dimensions.
        survival_rate: Float, used by drop connect.

    # Returns
        List: containing rotation estimates from every feature level.
    """
    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_dims * num_anchors]
    args = (subnet_repeats, num_filters, bias_initializer)
    initial_regressions = build_head(middles, *args, survival_rate,
                                     normalization='group')
    return refine_rotation_iteratively(*initial_regressions, subnet_iterations,
                                       *args, num_dims)


def refine_rotation_iteratively(rotation_features, initial_rotations,
                                subnet_iterations, subnet_repeats,
                                num_filters, bias_initializer, num_dims):
    """Iteratively refines rotation.

    # Arguments
        rotation_features: List, containing features from rotation head.
        initial_rotations: List, containing initial rotation values.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        num_dims: Int, number of pose dimensions.

    # Returns
        rotations: List, containing final rotation values from every
        feature level.
    """
    rotations = []
    iterator = zip(rotation_features, initial_rotations)
    args = (subnet_repeats, num_filters, bias_initializer)
    for rotation_feature, initial_rotation in iterator:
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([rotation_feature, initial_rotation])
            delta_rotation = refine_rotation(x, *args)
            initial_rotation = Add()([initial_rotation, delta_rotation])
        rotation = Reshape((-1, num_dims))(initial_rotation)
        rotations.append(rotation)
    return rotations


def refine_rotation(x, repeats, num_filters, bias_initializer,
                    channels_per_group=16):
    """Builds rotation refinement module.

    # Arguments
        x: Tensor, BiFPN layer output.
        repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        channels_per_group: Int, number of channels per group
            of Batchnormalization.

    # Returns
        delta_rotation: Tensor, after repeated convolution,
            group normalization and activation.
    """
    conv_body = build_head_conv2D(repeats, num_filters[0], bias_initializer)
    conv_head = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    num_groups = int(num_filters[0] / channels_per_group)
    for block_arg in range(repeats):
        x = conv_body[block_arg](x)
        x = GroupNormalization(groups=num_groups)(x)
        x = tf.nn.swish(x)
    return conv_head(x)


def TranslationNet(middles, subnet_iterations, subnet_repeats,
                   num_anchors, num_filters):
    """Initializes TranslationNet.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_anchors: List, number of combinations of anchor box's scale
            and aspect ratios.
        num_filters: Int, number of subnet filters.

    # Returns
        List: containing translation estimates from every feature level.
    """
    bias_initializer = tf.zeros_initializer()
    num_filters = [num_filters, num_anchors * 2, num_anchors]
    args = (subnet_repeats, num_filters, bias_initializer)
    initial_regressions = regress_initial_translations(middles, *args)
    return refine_translation_iteratively(*initial_regressions,
                                          *args, subnet_iterations)


def regress_initial_translations(middles, subnet_repeats, num_filters,
                                 bias_initializer):
    """Builds TranslationNet head.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.

    # Returns
        List: Containing initial_features, initial_xy and initial_z.
    """
    initial_features, initial_xy, initial_z = [], [], []
    args = (subnet_repeats, num_filters, bias_initializer)
    for x in middles:
        initial_translations = build_translation_subnets(x, *args)
        x, initial_translation_xy, initial_translation_z = initial_translations
        initial_features.append(x)
        initial_xy.append(initial_translation_xy)
        initial_z.append(initial_translation_z)
    return [initial_features, initial_xy, initial_z]


def build_translation_subnets(x, repeats, num_filters, bias_initializer,
                              channels_per_group=16):
    """Builds TranslationNet head.

    # Arguments
        x: Tensor, BiFPN layer output.
        repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        channels_per_group: Int, number of channels per group
            of Batchnormalization.

    # Returns
        List: Containing x, initial_xy and initial_z.
    """
    conv_body = build_head_conv2D(repeats, num_filters[0], bias_initializer)
    conv_head_xy = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    conv_head_z = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    num_groups = int(num_filters[0] / channels_per_group)
    for block_arg in range(repeats):
        x = conv_body[block_arg](x)
        x = GroupNormalization(groups=num_groups)(x)
        x = tf.nn.swish(x)
    return [x, conv_head_xy(x), conv_head_z(x)]


def refine_translation_iteratively(translation_features, translations_xy,
                                   translations_z, subnet_repeats, num_filters,
                                   bias_initializer, subnet_iterations):
    """Refines translation iteratively.

    # Arguments
        translation_features: List, containing features
            from translation head.
        translations_xy: List, containing translations in XY directions
            from translation head.
        translations_z: List, containing translations in Z directions
            from translation head.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        subnet_iterations: Int, number of iterative refinement steps
            used in rotation and translation subnets.

    # Returns
        translations: List, containing final translation values
            from every feature level.
    """
    translations = []
    args = (subnet_repeats, num_filters, bias_initializer)
    iterator = zip(translation_features, translations_xy, translations_z)
    for translation_feature, translation_xy, translation_z in iterator:
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([translation_feature,
                                      translation_xy, translation_z])
            delta_translations = refine_translation(x, *args)
            delta_translation_xy, delta_translation_z = delta_translations
            translation_xy = Add()([translation_xy, delta_translation_xy])
            translation_z = Add()([translation_z, delta_translation_z])
        translation_xy = Reshape((-1, 2))(translation_xy)
        translation_z = Reshape((-1, 1))(translation_z)
        translation = Concatenate(axis=-1)([translation_xy, translation_z])
        translations.append(translation)
    return translations


def refine_translation(x, repeats, num_filters, bias_initializer,
                       channels_per_group=16):
    """Translation refinement module.

    # Arguments
        x: Tensor, BiFPN layer output.
        repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        channels_per_group: Int, number of channels per group
            of Batchnormalization.

    # Returns
        List: Containing delta_xy, and delta_z.
    """
    conv_body = build_head_conv2D(repeats, num_filters[0], bias_initializer)
    conv_head_xy = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    conv_head_z = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    num_groups = int(num_filters[0] / channels_per_group)
    for block_arg in range(repeats):
        x = conv_body[block_arg](x)
        x = GroupNormalization(groups=num_groups)(x)
        x = tf.nn.swish(x)
    return [conv_head_xy(x), conv_head_z(x)]
