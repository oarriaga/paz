from keras import Model
from keras.constraints import MaxNorm
from keras.initializers import HeNormal
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Input
from keras.layers import ReLU
from keras.layers import Reshape
from keras.utils import get_file


WEIGHT_PATH = "https://github.com/oarriaga/altamira-data/releases/download/v0.17/simple_baseline_paz_jax.weights.h5"  # fmt: skip


def SimpleBaseline(input_shape=(32,), num_keypoints=16, keypoints_dim=3,
                   hidden_dim=1024, num_blocks=2, weights="human36m"):
    validate_weights(weights)
    keypoints2D = Input(shape=input_shape, name="keypoints2D")
    x = build_dense(keypoints2D, hidden_dim, "linear1_")
    x = BatchNormalization(name="batch_normalization")(x)
    x = ReLU()(x)
    for block in range(num_blocks):
        x = build_residual_block(x, hidden_dim, block)
    x = build_dense(x, num_keypoints * keypoints_dim, "linear4_")
    keypoints3D = Reshape((num_keypoints, keypoints_dim))(x)
    model = Model(keypoints2D, keypoints3D, name="simple_baseline")
    load_weights(model, weights)
    return model


def build_residual_block(x, units, block):
    residual = x
    x = build_dense(x, units, f"linear2_{block}")
    x = BatchNormalization(name=f"batch_normalization1{block}")(x)
    x = ReLU()(x)
    x = build_dense(x, units, f"linear3_{block}")
    x = BatchNormalization(name=f"batch_normalization2{block}")(x)
    x = ReLU()(x)
    return Add()([x, residual])


def build_dense(x, units, name):
    initializer = HeNormal()
    return Dense(units, kernel_initializer=initializer,
                 bias_initializer=initializer,
                 kernel_constraint=MaxNorm(max_value=1), name=name)(x)


def load_weights(model, weights):
    if weights != "human36m":
        return
    filename = WEIGHT_PATH.rsplit("/", 1)[-1]
    weights_path = get_file(filename, WEIGHT_PATH, cache_subdir="paz/models")
    print("Loading %s model weights" % weights_path)
    model.load_weights(weights_path)


def validate_weights(weights):
    if weights not in [None, "human36m"]:
        raise ValueError(f"Invalid weights: {weights}")
