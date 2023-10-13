import random

from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.configs import movinet as cfg
from official.projects.movinet.modeling import movinet
import tensorflow as tf

keras = tf.keras
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Flatten, Dense, LSTM, TimeDistributed)
from keras.applications.mobilenet import MobileNet
import random

def MoViNet(weights=None, input_shape=(38, 96, 96, 3), seed=305865):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: String, path to the weights file to load. TODO add weights implementation when weights are available
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)


        Model params according to vvadlrs3.pretrained_models.getFaceImageModel().summary()
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    # random.seed(seed)
    # initializer_glorot_lstm = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_dense = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_output = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_orthogonal = tf.keras.initializers.Orthogonal(seed=random.randint(0, 1000000))

    model_id = 'a0'

    # TODO do I need this? tf.keras.backend.clear_session()

    backbone = movinet.Movinet(model_id=model_id)
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=2)

    return model

def MoViNetManuall(weights=None, input_shape=(38, 96, 96, 3), seed=305865):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: String, path to the weights file to load. TODO add weights implementation when weights are available
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)


        Model params according to vvadlrs3.pretrained_models.getFaceImageModel().summary()
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    # random.seed(seed)
    # initializer_glorot_lstm = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_dense = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_output = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_orthogonal = tf.keras.initializers.Orthogonal(seed=random.randint(0, 1000000))

    # input_specs_dict = {'image': input_specs}
    # backbone = backbones.factory.build_backbone(
    #     input_specs=input_specs,
    #     backbone_config=model_config.backbone,
    #     norm_activation_config=model_config.norm_activation,
    #     l2_regularizer=l2_regularizer)
    # model = MovinetClassifier(
    #     backbone,
    #     num_classes=num_classes,
    #     kernel_regularizer=l2_regularizer,
    #     input_specs=input_specs_dict,
    #     activation=model_config.activation,
    #     dropout_rate=model_config.dropout_rate,
    #     output_states=model_config.output_states)

    return model


def MoViNetTut(weights=None, input_shape=(38, 96, 96, 3), seed=305865):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: String, path to the weights file to load. TODO add weights implementation when weights are available
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)


        Model params according to vvadlrs3.pretrained_models.getFaceImageModel().summary()
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    # random.seed(seed)
    # initializer_glorot_lstm = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_dense = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_output = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_orthogonal = tf.keras.initializers.Orthogonal(seed=random.randint(0, 1000000))

    # input_specs_dict = {'image': input_specs}

    model_id = 'a0'
    use_positional_encoding = model_id in {'a3', 'a4', 'a5'}

    backbone = movinet.Movinet(
        model_id=model_id,
        causal=True,
        conv_type='2plus1d',
        se_type='2plus3d',
        activation='hard_swish',
        gating_activation='hard_sigmoid',
        use_positional_encoding=use_positional_encoding,
        use_external_states=True,
    )

    model = movinet_model.MovinetClassifier(
        backbone,
        num_classes=2,
        output_states=True)

    return model
