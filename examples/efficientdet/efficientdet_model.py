from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import efficientnet_builder
from anchors import get_prior_boxes
from efficientdet_blocks import ResampleFeatureMap
from efficientdet_blocks import FPNCells, ClassNet, BoxNet
from utils import create_multibox_head

WEIGHT_PATH = (
    '/media/deepan/externaldrive1/project_repos/paz_versions'
    '/paz_efficientdet_weights/')


def EfficientDet(num_classes, base_weights, head_weights, input_shape,
                 fpn_num_filters, fpn_cell_repeats, box_class_repeats,
                 anchor_scale, min_level, max_level, fpn_weight_method,
                 return_base, model_name, backbone, training=False,
                 activation='swish', fpn_name='BiFPN', num_scales=3,
                 aspect_ratios=[1.0, 2.0, 0.5],
                 conv_batchnorm_activation_block=False,
                 use_batchnorm_for_sampling=True, conv_after_downsample=False,
                 separable_conv=True, survival_rate=None, name=""):
    """
    EfficientDet model in PAZ.
    # References
        -[Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)

    Initializes EfficientDet model.

    # Arguments
        image_size: Int, size of the input image.
        num_classes: Int, specifying the number of class in the
        output.
        fpn_num_filters: Int, FPN filter output size.
        fpn_cell_repeats: Int, Number of consecutive FPN block.
        box_class_repeats: Int, Number of consective regression
        and classification blocks.
        anchor_scale: Int, specifying the number of anchor
        scales.
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn.
        return_base: Bool, indicating the usage of features only
        from EfficientDet
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        training: Bool, whether EfficientDet architecture is trained.
        activation: A string specifying the activation function.
        fpn_name: A string specifying the feature fusion FPN
        layer.
        num_scales: Int, specifying the number of scales in the
        anchor boxes.
        aspect_ratios: List, specifying the aspect ratio of the
        conv_batchnorm_activation_block: Bool, specifying the presence
        of convolution - batch normalization - activation function
        patter in the EfficientDet building blocks.
        use_batchnorm_for_sampling: Bool, specifying the presense
        of batch normalization for resampling layers in feature
        extaction.
        conv_after_downsample: Bool, specifying the presence of
        convolution layer after downsampling.
        separable_conv: Bool, specifying the usage of separable
        convolution layers in EfficientDet.
        survival_rate: Float, specifying the survival probability
        name: Module name
    # Returns
        model: EfficientDet model specified in model_name with the following:
        class_outputs: Tensor, Logits for all classes corresponding to
        the features associated with the box coordinates.
        box_outputs: Tensor,  Box coordinate offsets for the
        corresponding prior boxes.
    """
    if base_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', base_weights)
    if head_weights not in ['COCO', None]:
        raise ValueError('Invalid base_weights: ', head_weights)
    if (base_weights == 'COCO') and (head_weights is None):
        raise NotImplementedError('Invalid `base_weights` with head_weights')
    if (base_weights is None) and (head_weights == 'COCO'):
        raise NotImplementedError('Invalid `base_weights` with head_weights')

    image = Input(shape=input_shape, name='image')
    num_levels = max_level - min_level + 1
    num_anchors = len(aspect_ratios) * num_scales
    num_filters = fpn_num_filters
    branch_tensors = efficientnet_builder.build_backbone(
        backbone, activation, survival_rate)(image)
    feature_levels = branch_tensors[min_level:max_level + 1]
    for level in range(6, max_level + 1):
        feature_levels.append(ResampleFeatureMap(
            (level - min_level), fpn_num_filters,
            use_batchnorm_for_sampling, conv_after_downsample,
            name='resample_p%d' % level)(feature_levels[-1], training, None))
    fpn_features = FPNCells(
        fpn_name, min_level, max_level, fpn_weight_method, fpn_cell_repeats,
        fpn_num_filters, use_batchnorm_for_sampling, conv_after_downsample,
        conv_batchnorm_activation_block, separable_conv,
        activation)(feature_levels, training)
    class_outputs = ClassNet(
        num_classes, num_anchors, num_filters,
        min_level, max_level, box_class_repeats,
        separable_conv, survival_rate)(fpn_features, training)
    box_outputs = BoxNet(
        num_anchors, num_filters, min_level,
        max_level, box_class_repeats, separable_conv,
        survival_rate)(fpn_features, training)
    branch_tensors = [class_outputs, box_outputs]
    if return_base:
        outputs = branch_tensors
    else:
        outputs = create_multibox_head(branch_tensors, num_levels, num_classes)
    model = Model(inputs=image, outputs=outputs, name=model_name)

    if (base_weights == 'COCO') and (head_weights == 'COCO'):
        weights_path = WEIGHT_PATH + model_name + '.h5'
        model.load_weights(weights_path)
    model.prior_boxes = get_prior_boxes(
        min_level, max_level, num_scales, aspect_ratios, anchor_scale,
        input_shape[0])
    return model
