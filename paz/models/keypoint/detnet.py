import numpy as np

from keras import Model
from keras import ops
from keras.initializers import TruncatedNormal
from keras.initializers import VarianceScaling
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Layer
from keras.layers import MaxPool2D
from keras.layers import ReLU
from keras.layers import Reshape
from keras.regularizers import l2
from keras.utils import get_file


def DetNet(input_shape=(128, 128, 3), num_keypoints=21, weights=None):
    validate_weights(weights)
    image = Input(shape=input_shape, dtype="uint8", name="image")
    x = ops.cast(image, "float32") / 255.0

    x = build_resnet50(x)
    pose_tile = PoseTile(name="pose_grid")(x)
    x = Concatenate(axis=-1, name="pose_merge")([x, pose_tile])

    heat_map = build_head_2D(x, num_keypoints, "heatmap0")
    x = Concatenate(axis=-1, name="heatmap0_merge")([x, heat_map])

    delta_map = build_head_3D(x, num_keypoints, "delta0")
    reshaped_delta_map = Reshape((32, 32, num_keypoints * 3), name="delta0_reshape")(delta_map)  # fmt: skip
    x = Concatenate(axis=-1, name="delta0_merge")([x, reshaped_delta_map])

    location_map = build_head_3D(x, num_keypoints, "location0")

    uv = compute_heatmap_uv(heat_map)
    xyz = compute_keypoint_xyz(location_map, uv)
    xyz = ops.take(xyz, 0, axis=0)
    uv = ops.take(uv, 0, axis=0)
    model = Model(image, [xyz, uv], name="detnet")
    load_weights(model, weights, "detnet")
    return model


def build_resnet50(x):
    x = build_conv_block(x, 64, 7, 2, "stem")
    x = build_stage(x, 256, 3, "stage2", last_stride=2)
    x = build_stage(x, 512, 4, "stage3", rate=2)
    x = build_stage(x, 1024, 6, "stage4", rate=4)
    return build_conv_block(x, 256, 3, 1, "squeeze")


def build_stage(x, filters, num_units, stage_name, last_stride=1, rate=1):
    for unit_arg in range(num_units):
        unit_name = name(stage_name, f"unit{unit_arg + 1}")
        strides = 1 if unit_arg < num_units - 1 else last_stride
        x = build_bottleneck(x, filters, strides, unit_name, rate)
    return x


def build_head_kwargs(string):
    keys = ["strides", "padding", "kernel_initializer", "name"]
    vals = {1, "same", TruncatedNormal(stddev=0.01), name(string, "output")}
    return dict(zip(keys, vals))


def build_head_2D(x, num_keypoints, head_name):
    x = build_conv_block(x, 256, 3, 1, name(head_name, "projection"))
    kwargs = build_head_kwargs(head_name)
    return Conv2D(num_keypoints, 1, activation="sigmoid", **kwargs)(x)


def build_head_3D(x, num_keypoints, head_name):
    x = build_conv_block(x, 256, 3, 1, name(head_name, "projection"))
    kwargs = build_head_kwargs(head_name)
    x = Conv2D(num_keypoints * 3, 1, **kwargs)(x)
    height, width = x.shape[1:3]
    return Reshape((height, width, num_keypoints, 3), name=name(head_name, "map"))(x)  # fmt: skip


def build_bottleneck(x, filters, strides, block_name, rate=1):
    num_channels = x.shape[-1]
    if num_channels == filters:
        if strides == 1:
            shortcut = x
        else:
            shortcut = MaxPool2D(pool_size=strides, strides=strides, padding="same", name=name(block_name, "shortcut_pool"))(x)  # fmt: skip
    else:
        shortcut = build_conv_block(x, filters, 1, strides, name(block_name, "shortcut"), with_relu=False)  # fmt: skip

    x = build_conv_block(x, filters // 4, 1, 1, name(block_name, "first"))
    x = build_conv_block(x, filters // 4, 3, strides, name(block_name, "second"), rate)  # fmt: skip
    x = build_conv_block(x, filters, 1, 1, name(block_name, "third"), with_relu=False)  # fmt: skip
    return ReLU(name=name(block_name, "output"))(shortcut + x)


def build_conv_block(x, filters, kernel_size, strides, block_name, rate=1, with_relu=True):  # fmt: skip
    conv_kwargs = {
        "filters": filters,
        "kernel_size": kernel_size,
        "strides": strides,
        "use_bias": False,
        "dilation_rate": rate,
        "kernel_regularizer": l2(0.5),
        "name": name(block_name, "conv"),
        "kernel_initializer": VarianceScaling(
            mode="fan_avg", distribution="uniform"
        ),
    }
    if strides == 1:
        x = Conv2D(padding="same", **conv_kwargs)(x)
    else:
        pad_before = (kernel_size - 1) // 2
        pad_after = (kernel_size - 1) - pad_before
        x = zero_pad(x, pad_before, pad_after)
        x = Conv2D(padding="valid", **conv_kwargs)(x)
    x = BatchNormalization(name=name(block_name, "batch_normalization"))(x)
    if with_relu:
        x = ReLU(name=name(block_name, "activation"))(x)
    return x


def compute_keypoint_xyz(location_map, uv):
    location_map = ops.transpose(location_map, [0, 3, 1, 2, 4])
    num_keypoints, height, width = location_map.shape[1:4]
    flat_index = uv[..., 0] * width + uv[..., 1]
    location_map = ops.reshape(location_map, [-1, num_keypoints, height * width, 3])  # fmt: skip
    flat_index = ops.expand_dims(flat_index, axis=-1)
    flat_index = ops.expand_dims(flat_index, axis=-1)
    flat_index = ops.concatenate([flat_index, flat_index, flat_index], axis=-1)
    location_map = ops.take_along_axis(location_map, flat_index, axis=2)
    return ops.squeeze(location_map, axis=2)


def compute_heatmap_uv(heat_map):
    _, height, width, num_keypoints = heat_map.shape
    heat_map = ops.reshape(heat_map, [-1, height * width, num_keypoints])
    argmax = ops.cast(ops.argmax(heat_map, axis=1), "int32")
    argmax_x = argmax // width
    argmax_y = argmax % width
    return ops.stack([argmax_x, argmax_y], axis=-1)


def zero_pad(tensor, pad_before, pad_after):
    return ops.pad(
        tensor,
        [
            [0, 0],
            [pad_before, pad_after],
            [pad_before, pad_after],
            [0, 0],
        ],
    )


def name(*parts):
    return "_".join(part for part in parts if part)


def load_weights(model, weights, model_name):
    if weights is None:
        return
    URL = "https://github.com/oarriaga/altamira-data/releases/download/v0.14/detnet_weights.hdf5"  # fmt: skip
    filename = URL.rsplit("/", 1)[-1]
    weights_path = get_file(filename, URL, cache_subdir="paz/models")
    print("Loading %s model weights" % weights_path)
    model.load_weights(weights_path)


def validate_weights(weights):
    if weights not in [None, "detnet"]:
        raise ValueError(f"Invalid weights: {weights}")


class PoseTile(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        axis = np.linspace(-1.0, 1.0, 32, dtype="float32")
        pose_grid = np.stack(
            [
                np.tile(axis.reshape(1, 32), [32, 1]),
                np.tile(axis.reshape(32, 1), [1, 32]),
            ],
            axis=-1,
        )
        self.pose_grid = ops.convert_to_tensor(pose_grid[None, ...])

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.broadcast_to(self.pose_grid, [batch_size, 32, 32, 2])
