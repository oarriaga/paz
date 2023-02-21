from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.constraints import MaxNorm
import tensorflow as tf
from tensorflow.keras.layers import (
	BatchNormalization,
	ReLU,
	Dense,
	Dropout,
	Input,
	Reshape
)


def dense_block(input_x, num_keypoints, rate):
	"""Make a bi-linear block with optional residual connection
	# Arguments
		input_x: the batch that enters the block
		linear_size: integer. The size of the linear units
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


def SIMPLE_BASELINE(num_keypoints, keypoints_dim, hidden_dim, input_shape,
                    num_layers, rate):
	"""keypoints model
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
	x = tf.squeeze(Reshape((num_keypoints, keypoints_dim))(x))
	model = Model(inputs, outputs=x)
	return model
