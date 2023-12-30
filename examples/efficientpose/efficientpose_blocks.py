import tensorflow as tf
from tensorflow.keras.layers import (GroupNormalization, Concatenate,
                                     Add, Reshape)
from paz.models.detection.efficientdet.efficientdet_blocks import (
    build_head_conv2D, build_head)


def build_pose_estimator_head(middles, subnet_iterations, subnet_repeats,
                              num_anchors, num_filters, num_dims):
    """Builds EfficientPose pose estimator's head.
    The built head includes RotationNet and TranslationNet
    for estimating rotation and translation respectively.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_iterations: Int, number of iterative refinement
            steps used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_anchors: List, number of combinations of
            anchor box's scale and aspect ratios.
        num_filters: Int, number of subnet filters.
        num_dims: Int, number of pose dimensions.

    # Returns
        List: Containing estimated rotations and translations of shape
        `(None, num_boxes, num_dims)` and
        `(None, num_boxes, num_dims)` respectively.
    """
    args = (middles, subnet_iterations, subnet_repeats, num_anchors)
    rotations = RotationNet(*args, num_filters, num_dims)
    rotations = Concatenate(axis=1,)(rotations)
    translations = TranslationNet(*args, num_filters)
    translations = Concatenate(axis=1)(translations)
    concatenate_transformation = Concatenate(axis=-1, name='transformation')
    transformations = concatenate_transformation([rotations, translations])
    return transformations


def RotationNet(middles, subnet_iterations, subnet_repeats,
                num_anchors, num_filters, num_dims, survival_rate=None):
    """Initializes RotationNet.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_iterations: Int, number of iterative refinement
            steps used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_anchors: List, number of combinations of
            anchor box's scale and aspect ratios.
        num_filters: Int, number of subnet filters.
        num_dims: Int, number of pose dimensions.

    # Returns
        List: containing rotation estimates from every feature level.
    """
    num_filters = [num_filters, num_dims * num_anchors]
    bias_initializer = tf.zeros_initializer()
    args = (subnet_repeats, num_filters, bias_initializer)
    rotations = build_head(middles, subnet_repeats, num_filters,
                           survival_rate, bias_initializer)
    return build_iterative_rotation_subnet(*rotations, subnet_iterations,
                                           *args, num_dims)


def build_iterative_rotation_subnet(rotation_features, initial_rotations,
                                    subnet_iterations, subnet_repeats,
                                    num_filters, bias_initializer, num_dims):
    """Builds iterative rotation subnets.

    # Arguments
        rotation_features: List, containing features from rotation head.
        initial_rotations: List, containing initial rotation values.
        subnet_iterations: Int, number of iterative refinement
            steps used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        num_dims: Int, number of pose dimensions.
        gn_groups: Int, number of groups in group normalization.
        gn_axis: Int, group normalization axis.

    # Returns
        rotations: List, containing final rotation values.
    """
    conv_blocks = build_head_conv2D(subnet_repeats - 1, num_filters[0],
                                    bias_initializer)
    head_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    args = (conv_blocks, subnet_repeats - 1)
    rotations = []
    for x, initial_rotation in zip(rotation_features, initial_rotations):
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([x, initial_rotation])
            x = refine_rotation(x, *args)
            delta_rotation = head_conv(x)
            initial_rotation = Add()([initial_rotation, delta_rotation])
        rotation = Reshape((-1, num_dims))(initial_rotation)
        rotations.append(rotation)
    return rotations


def refine_rotation(x, conv_blocks, repeats):
    """Rotation refinement module. Builds group normalization blocks
    followed by activation.

    # Arguments
        x: Tensor, BiFPN layer output.
        conv_blocks: List, containing convolutional blocks.
        repeats: Int, number of layers used in subnetworks.
        gn_groups: Int, number of groups in group normalization.
        gn_axis: Int, group normalization axis.

    # Returns
        x: Tensor, after repeated convolution,
            group normalization and activation.
    """
    for block_arg in range(repeats):
        x = conv_blocks[block_arg](x)
        x = GroupNormalization()(x)
        x = tf.nn.swish(x)
    return x


def TranslationNet(middles, subnet_iterations, subnet_repeats,
                   num_anchors, num_filters, survival_rate=None):
    """Initializes TranslationNet.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_iterations: Int, number of iterative refinement
            steps used in rotation and translation subnets.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_anchors: List, number of combinations of
            anchor box's scale and aspect ratios.
        num_filters: Int, number of subnet filters.

    # Returns
        List: containing translation estimates from every feature level.
    """
    num_filters = [num_filters, num_anchors * 2, num_anchors]
    bias_initializer = tf.zeros_initializer()
    args = (subnet_repeats, num_filters, bias_initializer)
    translations = build_translation_head(middles, *args)
    return build_iterative_translation_subnet(*translations, *args,
                                              subnet_iterations)


def build_translation_head(middles, subnet_repeats, num_filters,
                           bias_initializer):
    """Builds TranslationNet head.

    # Arguments
        middles: List, BiFPN layer output.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        gn_groups: Int, number of groups in group normalization.
        gn_axis: Int, group normalization axis.

    # Returns
        List: Containing translation_features,
            translations_xy and translations_z.
    """
    conv_blocks = build_head_conv2D(subnet_repeats, num_filters[0],
                                    bias_initializer)
    head_xy_conv = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    head_z_conv = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    args = (conv_blocks, subnet_repeats)
    translation_features, translations_xy, translations_z = [], [], []
    for x in middles:
        x = refine_rotation(x, *args)
        translation_xy = head_xy_conv(x)
        translation_z = head_z_conv(x)
        translation_features.append(x)
        translations_xy.append(translation_xy)
        translations_z.append(translation_z)
    return [translation_features, translations_xy, translations_z]


def build_iterative_translation_subnet(translation_features, translations_xy,
                                       translations_z, subnet_repeats,
                                       num_filters, bias_initializer,
                                       subnet_iterations):
    """Builds iterative translation subnets.

    # Arguments
        translation_features: List, containing
            features from translation head.
        translations_xy: List, containing translations
            in XY directions from translation head.
        translations_z: List, containing translations
            in Z directions from translation head.
        subnet_repeats: Int, number of layers used in subnetworks.
        num_filters: Int, number of subnet filters.
        bias_initializer: Callable, bias initializer.
        subnet_iterations: Int, number of iterative refinement
            steps used in rotation and translation subnets.
        gn_groups: Int, number of groups in group normalization.
        gn_axis: Int, group normalization axis.

    # Returns
        translations: List, containing final translation values.
    """
    conv_blocks = build_head_conv2D(subnet_repeats - 1, num_filters[0],
                                    bias_initializer)
    head_xy = build_head_conv2D(1, num_filters[1], bias_initializer)[0]
    head_z = build_head_conv2D(1, num_filters[2], bias_initializer)[0]
    args = (conv_blocks, subnet_repeats - 1)
    translations = []
    iterator = zip(translation_features, translations_xy, translations_z)
    for x, translation_xy, translation_z in iterator:
        for _ in range(subnet_iterations):
            x = Concatenate(axis=-1)([x, translation_xy, translation_z])
            x = refine_rotation(x, *args)
            delta_translation_xy = head_xy(x)
            delta_translation_z = head_z(x)
            translation_xy = Add()([translation_xy, delta_translation_xy])
            translation_z = Add()([translation_z, delta_translation_z])
        translation_xy = Reshape((-1, 2))(translation_xy)
        translation_z = Reshape((-1, 1))(translation_z)
        translation = Concatenate(axis=-1)([translation_xy, translation_z])
        translations.append(translation)
    return translations
