import tensorflow as tf
import efficientnet_builder
from anchors import get_prior_boxes
from efficientdet_blocks import ResampleFeatureMap
from efficientdet_blocks import FPNCells, ClassNet, BoxNet
from utils import create_multibox_head


class EfficientDet(tf.keras.Model):
    """
    EfficientDet model in PAZ.
    # References
        -[Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """

    def __init__(self, image_size, num_classes, fpn_num_filters,
                 fpn_cell_repeats, box_class_repeats, anchor_scale,
                 min_level, max_level, fpn_weight_method, return_base,
                 model_name, backbone, activation='swish', fpn_name='BiFPN',
                 num_scales=3, aspect_ratios=[1.0, 2.0, 0.5],
                 conv_batchnorm_activation_block=False,
                 use_batchnorm_for_sampling=True, conv_after_downsample=False,
                 separable_conv=True, survival_rate=None, name=""):
        """ Initializes EfficientDet model.

        # Arguments
            model_name: A string of EfficientDet model name.
            backbone: A string of EfficientNet backbone name used
            in EfficientDet.
            image_size: Int, size of the input image.
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
            activation: A string specifying the activation function.
            fpn_name: A string specifying the feature fusion FPN
            layer.
            num_classes: Int, specifying the number of class in the
            output.
            num_scales: Int, specifying the number of scales in the
            anchor boxes.
            aspect_ratios: List, specifying the aspect ratio of the
            default anchor boxes. Computed with k-mean on COCO dataset.
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
            return_base: Bool, indicating the usage of features only
            from EfficientDet
            name: Module name

        # Returns
            model: EfficientDet model specified in model_name
        """
        super().__init__(name=name)
        self.model_name = model_name
        self.backbone = backbone
        self.image_size = image_size
        self.fpn_num_filters = fpn_num_filters
        self.fpn_cell_repeats = fpn_cell_repeats
        self.box_class_repeats = box_class_repeats
        self.anchor_scale = anchor_scale
        self.min_level = min_level
        self.max_level = max_level
        self.fpn_weight_method = fpn_weight_method
        self.num_levels = max_level - min_level + 1
        self.activation = activation
        self.fpn_name = fpn_name
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.conv_batchnorm_activation_block = conv_batchnorm_activation_block
        self.use_batchnorm_for_sampling = use_batchnorm_for_sampling
        self.conv_after_downsample = conv_after_downsample
        self.separable_conv = separable_conv
        self.survival_rate = survival_rate
        self.return_base = return_base
        self.num_levels = max_level - min_level + 1
        self.prior_boxes = get_prior_boxes(
            min_level, max_level, num_scales, aspect_ratios, anchor_scale,
            image_size)
        self.backbone = efficientnet_builder.build_backbone(
            self.backbone, self.activation, self.survival_rate)
        self.resample_layers = []
        for level in range(6, self.max_level + 1):
            self.resample_layers.append(ResampleFeatureMap(
                (level - self.min_level), self.fpn_num_filters,
                self.use_batchnorm_for_sampling, self.conv_after_downsample,
                name='resample_p%d' % level))

        self.fpn_cells = FPNCells(
            self.fpn_name, self.min_level, self.max_level,
            self.fpn_weight_method, self.fpn_cell_repeats,
            self.fpn_num_filters, self.use_batchnorm_for_sampling,
            self.conv_after_downsample, self.conv_batchnorm_activation_block,
            self.separable_conv, self.activation)

        self.num_anchors = len(self.aspect_ratios) * self.num_scales
        self.num_filters = self.fpn_num_filters
        self.class_net = ClassNet(
            self.num_classes, self.num_anchors, self.num_filters,
            self.min_level, self.max_level, self.box_class_repeats,
            self.separable_conv, self.survival_rate)

        self.box_net = BoxNet(
            self.num_anchors, self.num_filters, self.min_level,
            self.max_level, self.box_class_repeats, self.separable_conv,
            self.survival_rate)

    def call(self, images, training=False):
        """Build EfficientDet model.

        # Arguments
            images: Tensor, indicating the image input to the architecture.
            training: Bool, whether EfficientDet architecture is trained.

        # Returns
            class_outputs: Tensor, Logits for all classes corresponding to
            the features associated with the box coordinates.
            box_outputs: Tensor,  Box coordinate offsets for the
            corresponding prior boxes.
        """

        # Efficientnet backbone features
        branch_tensors = self.backbone(images, training)

        feature_levels = branch_tensors[self.min_level:self.max_level + 1]

        # Build additional input features that are not from backbone.
        for resample_layer in self.resample_layers:
            feature_levels.append(resample_layer(
                feature_levels[-1], training, None))

        # BiFPN layers
        fpn_features = self.fpn_cells(feature_levels, training)

        # Classification head
        class_outputs = self.class_net(fpn_features, training)

        # Box regression head
        box_outputs = self.box_net(fpn_features, training)

        branch_tensors = [class_outputs, box_outputs]
        if self.return_base:
            outputs = branch_tensors
        else:
            outputs = create_multibox_head(branch_tensors,
                                           self.num_levels, self.num_classes)
        return outputs
