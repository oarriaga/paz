import argparse
import tensorflow as tf

import efficientnet_builder
from efficientdet_building_blocks import ResampleFeatureMap, \
    FPNCells, ClassNet, BoxNet

# Mock input image.
mock_input_image = tf.random.uniform((1, 224, 224, 3),
                                     dtype=tf.dtypes.float32,
                                     seed=1)


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
        self.backbone = efficientnet_builder.build_backbone(config)
        self.resample_layers = []
        for level in range(6, config["max_level"] + 1):
            self.resample_layers.append(ResampleFeatureMap(
                feature_level=(level - config["min_level"]),
                target_num_channels=config["fpn_num_filters"],
                apply_bn=config["apply_bn_for_resampling"],
                conv_after_downsample=config["conv_after_downsample"],
                data_format=config["data_format"],
                name='resample_p%d' % level,
            ))
        self.fpn_cells = FPNCells(config)

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
            data_format=config['data_format'],
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
            data_format=config['data_format'],
            feature_only=config['feature_only'],
        )

    def call(self, images, training=False):
        """Build EfficientDet model.
        # Arguments
            images: Tensor, indicating the image input to the architecture.
            training: Bool, whether EfficientDet architecture is trained.
        """

        # Efficientnet backbone features
        all_features = self.backbone(images)

        features = all_features[config["min_level"] - 1:
                                config["max_level"] + 1]

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


if __name__ == "__main__":

    description = "Build EfficientDet model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-m",
        "--model_name",
        default="EfficientDetD0",
        type=str,
        help="EfficientDet model name",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--backbone_name",
        default="EfficientNetB0",
        type=str,
        help="EfficientNet backbone name",
        required=False,
    )
    parser.add_argument(
        "-bw",
        "--backbone_weight",
        default="imagenet",
        type=str,
        help="EfficientNet backbone weight",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--act_type",
        default="swish",
        type=str,
        help="Activation function",
        required=False,
    )
    parser.add_argument(
        "--data_format",
        default="channels_last",
        type=str,
        help="Channels position in the data. "
        "Two options: channels_first and channels_last for "
        "(channels, image_height, image_width) or "
        "(image_height, image_width, channels), respectively",
        required=False,
    )
    parser.add_argument(
        "--min_level",
        default=3,
        type=int,
        help="EfficientNet feature minimum level. "
        "Level decides the activation map size, "
        "eg: For an input image of 640 x 640, "
        "the activation map resolution at level 3 is "
        "(640 / (2 ^ 3)) x (640 / (2 ^ 3))",
        required=False,
    )
    parser.add_argument(
        "--max_level",
        default=7,
        type=int,
        help="EfficientNet feature maximum level. "
        "Level decides the activation map size,"
        " eg: For an input image of 640 x 640, "
        "the activation map resolution at level 3 is"
        " (640 / (2 ^ 3)) x (640 / (2 ^ 3))",
        required=False,
    )
    parser.add_argument(
        "--fpn_name",
        default="BiFPN",
        type=str,
        help="Feature Pyramid Network name",
        required=False,
    )
    parser.add_argument(
        "--fpn_weight_method",
        default="fastattn",
        type=str,
        help="FPN weight method to fuse features. "
             "Options available: attn, fastattn",
        required=False,
    )
    parser.add_argument(
        "--fpn_num_filters",
        default=64,
        type=int,
        help="Number of filters at the FPN convolutions",
        required=False,
    )
    parser.add_argument(
        "--fpn_cell_repeats",
        default=3,
        type=int,
        help="Number of FPNs repeated in the FPN layer",
        required=False,
    )
    parser.add_argument(
        "--apply_bn_for_resampling",
        default=True,
        type=bool,
        help="Flag to apply batch normalization after resampling features",
        required=False,
    )
    parser.add_argument(
        "--conv_after_downsample",
        default=True,
        type=bool,
        help="Flag to apply convolution after downsampling features",
        required=False,
    )
    parser.add_argument(
        "--conv_bn_act_pattern",
        default=True,
        type=bool,
        help="Flag to apply convolution, batch normalization and activation",
        required=False,
    )
    parser.add_argument(
        "--separable_conv",
        default=True,
        type=bool,
        help="Flag to use separable convolutions",
        required=False,
    )
    parser.add_argument(
        "--aspect_ratios",
        default=[1.0, 2.0, 0.5],
        type=list,
        action='append',
        help="Aspect ratio of the boxes",
        required=False,
    )
    parser.add_argument(
        "--survival_prob",
        default=None,
        type=float,
        help="Survival probability for drop connect",
        required=False,
    )
    parser.add_argument(
        "--num_classes",
        default=90,
        type=int,
        help="Number of classes in the dataset",
        required=False,
    )
    parser.add_argument(
        "--num_scales",
        default=3,
        type=int,
        help="Number of scales for the boxes",
        required=False,
    )
    parser.add_argument(
        "--box_class_repeats",
        default=3,
        type=int,
        help="Number of repeated blocks in box and class net",
        required=False,
    )
    parser.add_argument(
        "--feature_only",
        default=False,
        type=bool,
        help="Whether feature only is required from EfficientDet",
        required=False,
    )

    args = parser.parse_args()
    config = vars(args)
    print(config)
    # TODO: Add parsed user-inputs to the config and update the config
    efficientdet = EfficientDet(config=config)
    class_outputs, box_outputs = efficientdet(mock_input_image, False)
