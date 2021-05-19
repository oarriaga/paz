import argparse
import tensorflow as tf
from efficientdet_model import EfficientDet

# Mock input image.
mock_input_image = tf.random.uniform((1, 224, 224, 3),
                                     dtype=tf.dtypes.float32,
                                     seed=1)


if __name__ == "__main__":

    description = "Build EfficientDet model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-m",
        "--model_name",
        default="efficientdetd0",
        type=str,
        help="EfficientDet model name",
        required=False,
    )
    parser.add_argument(
        "-b",
        "--backbone_name",
        default="efficientnetb0",
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
        "--use_batchnorm_for_sampling",
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
        "--conv_batchnorm_act_pattern",
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
    efficientdet.build(mock_input_image.shape)
    print(efficientdet.summary())
    class_outputs, box_outputs = efficientdet(mock_input_image, False)
