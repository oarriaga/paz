from efficientnet_model import EfficientNet


def get_efficientnet_params(model_name):
    """Default efficientnet scaling coefficients and
    image name based on model name.
    The value of each model name in the key represents:
    (width_coefficient, depth_coefficient, survival_rate).
    with_coefficient: scaling coefficient for network width.
    depth_coefficient: scaling coefficient for network depth.
    survival_rate: survival rate for final fully connected layers.

    # Arguments
        model_name: String, name of the EfficientNet backbone

    # Returns
        efficientnetparams: Dictionary, parameters corresponding to
        width coefficient, depth coefficient, survival rate
    """
    efficientnet_params = {'efficientnet-b0': (1.0, 1.0, 0.8),
                           'efficientnet-b1': (1.0, 1.1, 0.8),
                           'efficientnet-b2': (1.1, 1.2, 0.7),
                           'efficientnet-b3': (1.2, 1.4, 0.7),
                           'efficientnet-b4': (1.4, 1.8, 0.6),
                           'efficientnet-b5': (1.6, 2.2, 0.6),
                           'efficientnet-b6': (1.8, 2.6, 0.5),
                           'efficientnet-b7': (2.0, 3.1, 0.5),
                           'efficientnet-b8': (2.2, 3.6, 0.5),
                           'efficientnet-l2': (4.3, 5.3, 0.5)}
    return efficientnet_params[model_name]


def get_efficientnet_model(model_name):
    """Create a base feature network and return the features before pooling.

    # Arguments
        model_name: String, name of the EfficientNet backbone

    # Returns:
        model: EfficientNet model
    """
    efficientnet_param = get_efficientnet_params(model_name)
    width_coefficient, depth_coefficient, survival_rate = efficientnet_param
    model = EfficientNet(width_coefficient, depth_coefficient, survival_rate,
                         model_name)
    return model
