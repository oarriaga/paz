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


def IKNet(
    input_shape=(84, 3), num_keypoints=21, depth=6, width=1024, weights=None
):
    validate_weights(weights)
    keypoints = Input(shape=input_shape, name="keypoints")
    x = Reshape((1, -1), name="input_reshape")(keypoints)
    for depth_arg in range(depth):
        x = build_dense_block(x, width, depth_arg)
        x = Activation("sigmoid", name=f"sigmoid_activation_{depth_arg}")(x)

    quaternions = build_dense(x, num_keypoints * 4, "output")
    quaternions = Reshape((num_keypoints, 4), name="output_reshape")(quaternions)  # fmt: skip
    quaternions = normalize(quaternions)
    positive_mask = ops.tile(quaternions[:, :, 0:1] > 0, [1, 1, 4])
    quaternions = ops.where(positive_mask, quaternions, -quaternions)
    quaternions = reorder_quaternions(quaternions)
    model = Model(keypoints, [quaternions], name="iknet")
    if weights is not None:
        load_weights(model)
    return model


def build_dense_block(features, num_units, depth_arg):
    features = build_dense(features, num_units, f"dense_block_{depth_arg}")
    return BatchNormalization(name=f"batch_normalization_{depth_arg}")(features)


def build_dense(features, num_units, name):
    return Dense(
        num_units,
        kernel_regularizer=l2(0.5),
        kernel_initializer=TruncatedNormal(stddev=0.01),
        name=f"{name}_dense",
    )(features)


def normalize(values):
    values_norm = ops.sqrt(ops.sum(values * values, axis=-1, keepdims=True))
    values_norm = ops.maximum(values_norm, 1e-6)
    return values / values_norm


def reorder_quaternions(quaternions):
    scalar = quaternions[:, :, 0:1]
    vector = quaternions[:, :, 1:4]
    return ops.concatenate([vector, scalar], axis=-1)


def load_weights(model):
    URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.14/iknet_weight.hdf5"  # fmt: skip
    filename = URL.rsplit("/", 1)[-1]
    weights_path = get_file(filename, URL, cache_subdir="paz/models")
    print("Loading %s model weights" % weights_path)
    model.load_weights(weights_path)


def validate_weights(weights):
    if weights not in [None, "iknet"]:
        raise ValueError(f"Invalid weights: {weights}")
