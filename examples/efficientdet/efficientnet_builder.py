import tensorflow as tf
import efficientnet.tfkeras as efficientnet


def build_backbone(config):
    """
    Build backbone model.
    # Arguments
        config: Configuration of the EfficientDet model.
    # Returns
        EfficientNet model with intermediate feature levels.
    """

    backbone_name = config["backbone_name"]
    weight = config["backbone_weight"]

    efficientnet_class = getattr(efficientnet, backbone_name)
    backbone = efficientnet_class(weights=weight, include_top=False)
    backbone_layers = backbone.layers
    features = []
    for layer, next_layer in zip(backbone_layers[:-1], backbone_layers[1:]):
        if hasattr(next_layer, "strides") and next_layer.strides[0] == 2:
            features.append(layer)
    features.append(next_layer)

    return tf.keras.Model(
        backbone.input,
        outputs=[f.output for f in features[-5:]],
        name=config["backbone_name"],
    )
