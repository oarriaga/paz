import argparse
import tensorflow as tf


import hparams_config
import efficientnet_builder

# Mock input image.
mock_input_image = tf.random.uniform((1, 3, 240, 240), dtype=tf.dtypes.float32, seed=1)

class EfficientDet(tf.keras.Model):
    """
    EfficientDet model in PAZ.
    # References
        -[Google AutoML repository implementation of EfficientDet](
        https://github.com/google/automl/tree/master/efficientdet)
    """

    def __init__(self,
                 model_name=None,
                 config=None,
                 name='',
                 feature_only=False):
        """Initialize model.
        # Arguments
            model_name: String, indicating the EfficientDet-Dx architecture name, x denotes the EfficientDet type.
            config: Config class in hparams_config, will be refactored in the final version.
            name: A string of layer name.
            feature_only: Boolean, for building the base feature network only
        """
        super().__init__(name=name)

        config = config or hparams_config.get_efficientdet_config(model_name)
        self.config = config

        self.backbone = efficientnet_builder.build_backbone(config)

    def call(self, images, training=False):
        """Build EfficientDet model.
            # Arguments
                images: Tensor, indicating the image input to the architecture.
                training: Bool, indicating whether EfficientDet architecture is trained.
            """
        # Efficientnet backbone features
        all_features = self.backbone(images, training=training, features_only=True)

        # BiFPN layers
        # TODO: Implement BiFPN

        # Classification head
        # TODO: Implement classification head
        class_outputs = None

        # Box regression head
        # TODO: Implement box regression head
        box_outputs = None

        return class_outputs, box_outputs


if __name__ == "__main__":
    description = 'Build EfficientDet model'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-m',
        '--model_name',
        default='efficientdet-d0',
        type=str,
        help='EfficientDet model name',
        required=False,
    )
    parser.add_argument(
        '-b',
        '--backbone_name',
        default='efficientnet-b0',
        type=str,
        help='EfficientNet backbone name',
        required=False,
    )
    parser.add_argument(
        '-a',
        '--activation',
        default='swish',
        type=str,
        help='Activation function',
        required=False,
    )
    args = parser.parse_args()
    model_name = args.model_name
    backbone_name = args.backbone_name
    efficientdet = EfficientDet(model_name, config=None)
    class_outputs, box_outputs = efficientdet(mock_input_image, False)
