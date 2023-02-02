from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import (
    BatchNormalization,
    ReLU,
    Dense,
    Dropout,
    Input
)


def dense_block(input_x, num_keypoints, residual, rate, max_norm,
               batch_norm):
    """Make a bi-linear block with optional residual connection  
    # Arguments
        input_x: the batch that enters the block
        linear_size: integer. The size of the linear units
        residual: boolean. Whether to add a residual connection
        dropout_keep_prob: float [0,1]. Probability of dropping something out
        max_norm: boolean. Whether to clip weights to 1-norm
        batch_norm: boolean. Whether to do batch normalization       
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


def SIMPLE_BASELINE(num_keypoints, input_shape, num_layers, residual, 
                    max_norm, batch_norm, rate):
    HUMAN_3D_SIZE = 16 * 3
    inputs = Input(shape=input_shape)
    kwargs = {'kernel_initializer': HeNormal(), 'bias_initializer': HeNormal(),
              'kernel_constraint': MaxNorm(max_value=1)}
    x = Dense(num_keypoints, use_bias=True, **kwargs)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate)(x)
    for layer in range(num_layers):
        x = dense_block(x, num_keypoints, residual, rate, max_norm, batch_norm)
    x = Dense(HUMAN_3D_SIZE, **kwargs)(x)
    model = Model(inputs, outputs=x)
    return model
