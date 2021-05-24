import tensorflow as tf
import efficientnet_builder
from efficientdet_building_blocks import ResampleFeatureMap, \
    FPNCells, ClassNet, BoxNet


class EfficientDet(tf.keras.Model):
    """
    EfficientDet model in PAZ.
    # References
        -[Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """

    def __init__(self, config, name=""):
        """Initialize model.
        # Arguments
            config: Configuration of the EfficientDet model.
            name: A string of layer name.
        """
        super().__init__(name=name)

        self.config = config
        self.backbone = efficientnet_builder.build_backbone(
            backbone_name=config['backbone_name'],
            activation_fn=config['act_type'],
            survival_prob=config['survival_prob']
            )
        self.resample_layers = []
        for level in range(6, config["max_level"] + 1):
            self.resample_layers.append(ResampleFeatureMap(
                feature_level=(level - config["min_level"]),
                target_num_channels=config["fpn_num_filters"],
                use_batchnorm=config["use_batchnorm_for_sampling"],
                conv_after_downsample=config["conv_after_downsample"],
                name='resample_p%d' % level,
            ))

        self.fpn_cells = FPNCells(
            fpn_name=config['fpn_name'],
            min_level=config['min_level'],
            max_level=config['max_level'],
            fpn_weight_method=config['fpn_weight_method'],
            fpn_cell_repeats=config['fpn_cell_repeats'],
            fpn_num_filters=config['fpn_num_filters'],
            use_batchnorm_for_sampling=config['use_batchnorm_for_sampling'],
            conv_after_downsample=config['conv_after_downsample'],
            conv_batchnorm_act_pattern=config['conv_batchnorm_act_pattern'],
            separable_conv=config['separable_conv'],
            act_type=config['act_type'])

        num_anchors = len(config['aspect_ratios']) * config['num_scales']
        num_filters = config['fpn_num_filters']
        self.class_net = ClassNet(
            num_classes=config['num_classes'],
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config['min_level'],
            max_level=config['max_level'],
            act_type=config['act_type'],
            repeats=config['box_class_repeats'],
            separable_conv=config['separable_conv'],
            survival_prob=config['survival_prob'],
            feature_only=config['feature_only'],
        )

        self.box_net = BoxNet(
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config['min_level'],
            max_level=config['max_level'],
            act_type=config['act_type'],
            repeats=config['box_class_repeats'],
            separable_conv=config['separable_conv'],
            survival_prob=config['survival_prob'],
            feature_only=config['feature_only'],
        )

    def call(self, images, training=False):
        """Build EfficientDet model.
        # Arguments
            images: Tensor, indicating the image input to the architecture.
            training: Bool, whether EfficientDet architecture is trained.
        """

        # Efficientnet backbone features
        all_features = self.backbone(images,
                                     training=training,
                                     features_only=True)

        features = all_features[self.config["min_level"]:
                                self.config["max_level"] + 1]

        # Build additional input features that are not from backbone.
        for resample_layer in self.resample_layers:
            features.append(resample_layer(features[-1], training, None))

        # BiFPN layers
        fpn_features = self.fpn_cells(features, training)

        # Classification head
        class_outputs = self.class_net(fpn_features, training)

        # Box regression head
        box_outputs = self.box_net(fpn_features, training)

        return class_outputs, box_outputs
