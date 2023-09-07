import os
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape


def dense_block(input_x, num_keypoints, rate):
    """Make a bi-linear block with optional residual connection
    # Arguments
        input_x: the batch that enters the block
        num_keypoints: integer. The size of the linear units
        rate: float [0,1]. Probability of dropping something out

    # Returns
        x: the batch after it leaves the block
    """
    kwargs = {'kernel_initializer': HeNormal(), 'bias_initializer': HeNormal(),
              'kernel_constraint': MaxNorm(max_value=1)}
    x = Dense(num_keypoints, **kwargs)(input_x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate)(x)
    x = Dense(num_keypoints, **kwargs, )(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate)(x)
    x = (x + input_x)
    return x


def SimpleBaseline(input_shape=(32,), num_keypoints=16, keypoints_dim=3,
                   hidden_dim=1024, num_layers=2, rate=1, weights='human36m'):
    """Model that predicts 3D keypoints from 2D keypoints
    # Arguments
        num_keypoints: numer of kepoints
        keypoints_dim: dimension of keypoints
        hidden_dim: size of hidden layers
        input_shape: size of the input
        num_layers: number of layers
        rate: dropout drop rate

    # Returns
        keypoints3D estimation model
    """
    inputs = Input(shape=input_shape)
    kwargs = {'kernel_initializer': HeNormal(), 'bias_initializer': HeNormal(),
              'kernel_constraint': MaxNorm(max_value=1)}
    x = Dense(hidden_dim, use_bias=True, **kwargs)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate)(x)
    for layer in range(num_layers):
        x = dense_block(x, hidden_dim, rate)
    x = Dense(num_keypoints * keypoints_dim, **kwargs)(x)
    x = Reshape((num_keypoints, keypoints_dim))(x)
    model = Model(inputs, outputs=x)
    if weights == 'human36m':
        URL = ('https://github.com/oarriaga/altamira-data/releases/download/'
               'v0.17/SIMPLE-BASELINES.hdf5')
        filename = os.path.basename(URL)
        weights_path = get_file(filename, URL, cache_subdir='paz/models')
        model.load_weights(weights_path)
    return model
