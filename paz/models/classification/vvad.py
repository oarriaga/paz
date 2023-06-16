from vvadlrs3 import pretrained_models

def VVAD(weights=None):
    """Build MiniXception (see references).

    # Arguments
        input_shape: List of three integers e.g. ``[H, W, 3]``
        num_classes: Int.
        weights: ``None`` or string with pre-trained dataset. Valid datasets
            include only ``FER``.

    # Returns
        Tensorflow-Keras model.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """
    if weights == 'VVAD-LRS3':
        model = pretrained_models.getFaceImageModel()
    else:
        raise ValueError('Invalid model', weights)

    model._name = 'VVAD-LRS3'
    return model