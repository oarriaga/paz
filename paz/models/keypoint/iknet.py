from keras import Model
from keras import ops
from keras.initializers import TruncatedNormal
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.regularizers import l2
from keras.utils import get_file


WEIGHT_PATH = "https://github.com/oarriaga/altamira-data/releases/download/v0.14/iknet_paz_jax.weights.h5"  # fmt: skip


def IKNet(input_shape=(84, 3), num_keypoints=21, depth=6, width=1024,
          weights=None):
    validate_weights(weights)
    keypoints = Input(shape=input_shape, name="keypoints")
    features = build_encoder(keypoints, depth, width)
    quaternions = build_quaternions(features, num_keypoints)
    model = Model(keypoints, quaternions, name="iknet")
    if weights is not None:
        load_weights(model)
    return model


def build_encoder(keypoints, depth, width):
    features = Reshape((1, -1), name="input_reshape")(keypoints)
    for block_arg in range(depth):
        features = build_dense_block(features, width, block_arg)
    return features


def build_dense_block(features, num_units, block_arg):
    features = build_dense(features, num_units, f"dense_block_{block_arg}")
    batch_norm_name = f"batch_normalization_{block_arg}"
    features = BatchNormalization(name=batch_norm_name)(features)
    activation_name = f"sigmoid_activation_{block_arg}"
    return Activation("sigmoid", name=activation_name)(features)


def build_dense(features, num_units, block_name):
    initializer = TruncatedNormal(stddev=0.01)
    return Dense(num_units, kernel_regularizer=l2(0.5),
                 kernel_initializer=initializer,
                 name=f"{block_name}_dense")(features)


def build_quaternions(features, num_keypoints):
    quaternions = build_dense(features, num_keypoints * 4, "output")
    shape = (num_keypoints, 4)
    quaternions = Reshape(shape, name="output_reshape")(quaternions)
    quaternions = normalize(quaternions)
    quaternions = flip_to_positive_scalar(quaternions)
    return reorder_quaternions(quaternions)


def normalize(quaternions):
    squared_norm = ops.sum(quaternions * quaternions, axis=-1, keepdims=True)
    norm = ops.maximum(ops.sqrt(squared_norm), 1e-6)
    return quaternions / norm


def flip_to_positive_scalar(quaternions):
    positive = ops.tile(quaternions[:, :, 0:1] > 0, [1, 1, 4])
    return ops.where(positive, quaternions, -quaternions)


def reorder_quaternions(quaternions):
    scalar = quaternions[:, :, 0:1]
    vector = quaternions[:, :, 1:4]
    return ops.concatenate([vector, scalar], axis=-1)


def load_weights(model):
    filename = WEIGHT_PATH.rsplit("/", 1)[-1]
    weights_path = get_file(filename, WEIGHT_PATH, cache_subdir="paz/models")
    print("Loading %s model weights" % weights_path)
    model.load_weights(weights_path)


def validate_weights(weights):
    if weights not in [None, "iknet"]:
        raise ValueError(f"Invalid weights: {weights}")
