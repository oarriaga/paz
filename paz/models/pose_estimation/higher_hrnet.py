import h5py
import numpy as np

from keras import Model
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Input
from keras.layers import ReLU
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.utils import get_file


WEIGHT_PATH = "https://github.com/oarriaga/altamira-data/releases/download/v0.10/HigherHRNet_weights.hdf5"  # fmt: skip


def HigherHRNet(weights="COCO", input_shape=(None, None, 3), num_keypoints=17,
                with_AE_loss=(True, False)):
    validate_weights(weights)
    image = Input(shape=input_shape, name="image")
    x = build_stem(image, 64)
    x = build_stage1(x)
    branches = build_transition1(x)
    branches = build_stage2(branches)
    branches = build_stage3(branches)
    branches = build_stage4(branches)
    outputs = build_outputs(branches[0], num_keypoints, with_AE_loss)
    model = Model(image, outputs, name="HigherHRNet")
    load_weights(model, weights)
    return model


def build_stem(x, filters):
    x = ZeroPadding2D(padding=(1, 1), name="pad")(x)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name="conv1")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name="bn1")(x)
    x = ReLU(name="relu")(x)
    x = ZeroPadding2D(padding=(1, 1), name="pad_1")(x)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name="conv2")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name="bn2")(x)
    return ReLU()(x)


def build_stage1(x):
    x = bottleneck(x, 64, 4, downsample=True, name="layer1.0")
    for block in range(1, 4):
        x = bottleneck(x, 64, 4, name=f"layer1.{block}")
    return x


def build_transition1(x):
    full = Conv2D(32, 3, padding="same", use_bias=False,
                  name="transition1.0.0")(x)
    full = BatchNormalization(momentum=0.1, name="transition1.0.1")(full)
    full = ReLU()(full)
    half = transition_block(x, 2, name="transition1.1.")
    return [full, half]


def build_stage2(branches):
    branches = apply_modules(branches, stage=2, num_modules=1,
                             output_branches=2, name="stage2")
    branches.append(transition_block(branches[1], 2, name="transition2.2."))
    return branches


def build_stage3(branches):
    branches = apply_modules(branches, stage=3, num_modules=4,
                             output_branches=3, name="stage3")
    branches.append(transition_block(branches[2], 2, name="transition3.3."))
    return branches


def build_stage4(branches):
    for module in range(3):
        branches = apply_blocks(branches, stage=4, module=module, name="stage4")
        output_branches = 1 if module == 2 else 4
        name = f"stage4.{module}.fuse_layers"
        branches = fuse_layers(branches, 4, output_branches, name=name)
    return branches


def apply_modules(branches, stage, num_modules, output_branches, name):
    for module in range(num_modules):
        branches = apply_blocks(branches, stage, module, name)
        fuse_name = f"{name}.{module}.fuse_layers"
        branches = fuse_layers(branches, stage, output_branches, name=fuse_name)
    return branches


def apply_blocks(branches, stage, module, name):
    for block in range(4):
        block_name = f"{name}.{module}.branches.{block}"
        branches = blocks_in_branch(branches, stage, 32, block_name)
    return branches


def build_outputs(branch, num_keypoints, with_AE_loss):
    channels = num_keypoints * 2 if with_AE_loss[0] else num_keypoints
    output = Conv2D(channels, 1, padding="same", name="final_layers.0")(branch)
    x = Concatenate(axis=-1)([branch, output])
    x = deconv_layers(x, 32)
    channels = num_keypoints * 2 if with_AE_loss[1] else num_keypoints
    upsampled = Conv2D(channels, 1, padding="same", name="final_layers.1")(x)
    return [output, upsampled]


def bottleneck(tensor, filters, expansion, downsample=None, name=None):
    residual = tensor
    x = Conv2D(filters, 1, use_bias=False, name=name + ".conv1")(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + ".bn1")(x)
    x = ReLU(name=name + ".relu")(x)
    x = Conv2D(filters, 3, padding="same", use_bias=False,
               name=name + ".conv2")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + ".bn2")(x)
    x = ReLU()(x)
    x = Conv2D(filters * expansion, 1, use_bias=False, name=name + ".conv3")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + ".bn3")(x)
    if downsample is not None:
        residual = Conv2D(256, 1, use_bias=False,
                          name="layer1.0.downsample.0")(tensor)
        residual = BatchNormalization(momentum=0.1, epsilon=1e-05,
                                      name="layer1.0.downsample.1")(residual)
    x = Add()([x, residual])
    return ReLU()(x)


def basic_block(tensor, filters, name=None):
    x = Conv2D(filters, 3, padding="same", use_bias=False,
               name=name + ".conv1")(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + ".bn1")(x)
    x = ReLU(name=name + ".relu")(x)
    x = Conv2D(filters, 3, padding="same", use_bias=False,
               name=name + ".conv2")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + ".bn2")(x)
    x = Add()([x, tensor])
    return ReLU()(x)


def blocks_in_branch(tensors, stage, in_channels, name):
    for branch_arg in range(stage):
        filters = in_channels * (2 ** branch_arg)
        block_name = f"{name[:18]}{branch_arg}.{name[18:]}"
        tensors[branch_arg] = basic_block(tensors[branch_arg], filters,
                                          name=block_name)
    return tensors


def transition_block(tensor, alpha, name):
    in_channels = tensor.shape[-1]
    filters = 32 * alpha if in_channels == 256 else in_channels * alpha
    x = ZeroPadding2D(padding=(1, 1))(tensor)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name=name + "0.0")(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + "0.1")(x)
    return ReLU()(x)


def fuse_layers(tensors, stage, output_branches, filters=32, name=None):
    all_tensors = []
    for branch_arg in range(output_branches):
        x_to_y_tensors = []
        for stage_arg in range(stage):
            y = fuse_pair(tensors, branch_arg, stage_arg, filters, name)
            x_to_y_tensors.append(y)
        all_tensors.append(x_to_y_tensors)
    return reduce_fused_tensors(all_tensors)


def fuse_pair(tensors, branch_arg, stage_arg, filters, name):
    steps = stage_arg - branch_arg
    if steps == 0:
        return tensors[branch_arg]
    if steps > 0:
        upsample_name = ".".join((name, str(branch_arg), str(stage_arg)))
        size = (2 ** steps, 2 ** steps)
        return upsample(tensors[stage_arg], filters * (2 ** branch_arg),
                        size, name=upsample_name)
    return fuse_downsample(tensors, branch_arg, stage_arg, filters, name)


def fuse_downsample(tensors, branch_arg, stage_arg, filters, name):
    y, y_flag, iterations = None, False, 0
    for _ in range((-1 * steps_between(branch_arg, stage_arg)) - 1):
        iterations += 1
        if y_flag:
            names = get_names(name, branch_arg, stage_arg, 1.0)
            y = downsample(y, filters * (2 ** stage_arg), names)
        else:
            names = get_names(name, branch_arg, stage_arg, 0.0)
            y = downsample(tensors[stage_arg], filters * (2 ** stage_arg),
                           names, with_padding=False)
        y = ReLU()(y)
        y_flag = True
    if not y_flag:
        tensors[stage_arg] = ZeroPadding2D()(tensors[stage_arg])
        names = get_names(name, branch_arg, stage_arg, 0.0)
        return downsample(tensors[stage_arg], filters * (2 ** branch_arg),
                          names, with_padding=False)
    names = get_names(name, branch_arg, stage_arg, 0.0, iterations)
    return downsample(y, filters * (2 ** branch_arg), names)


def steps_between(branch_arg, stage_arg):
    return stage_arg - branch_arg


def reduce_fused_tensors(all_tensors):
    x_fused = []
    for x_arg in range(len(all_tensors)):
        for y_arg in range(len(all_tensors[x_arg])):
            if y_arg == 0:
                x_fused.append(all_tensors[x_arg][0])
            else:
                x_fused[x_arg] = Add()([x_fused[x_arg], all_tensors[x_arg][y_arg]])  # fmt: skip
    return [ReLU()(tensor) for tensor in x_fused]


def get_names(name, branch_arg, stage_arg, counter, iterations=0):
    base = (name, str(branch_arg), str(stage_arg))
    name1 = ".".join(base + (str(iterations + counter),))
    name2 = ".".join(base + (str(iterations + counter + 0.1),))
    return [name1, name2]


def upsample(tensor, filters, size, name=None):
    x = Conv2D(filters, 1, use_bias=False, name=name + ".0")(tensor)
    x = BatchNormalization(momentum=0.1, epsilon=1e-05, name=name + ".1")(x)
    return UpSampling2D(size=size, interpolation="nearest", name=name + ".2")(x)


def downsample(tensor, filters, name=None, with_padding=True):
    if with_padding:
        tensor = ZeroPadding2D(padding=(1, 1))(tensor)
    x = Conv2D(filters, 3, strides=2, use_bias=False, name=name[0])(tensor)
    return BatchNormalization(momentum=0.1, epsilon=1e-05, name=name[1])(x)


def deconv_layers(tensor, output_channels, num_deconv=1):
    for _ in range(num_deconv):
        x = Conv2DTranspose(output_channels, 4, strides=2, padding="same",
                            use_bias=False, name="deconv_layers.0.0.0")(tensor)
        x = BatchNormalization(momentum=0.1, epsilon=1e-05,
                               name="deconv_layers.0.0.1")(x)
        x = ReLU()(x)
        for block in range(4):
            name = f"deconv_layers.0.{block + 1}.0"
            x = basic_block(x, output_channels, name=name)
    return x


def load_weights(model, weights):
    if weights != "COCO":
        return
    filename = WEIGHT_PATH.rsplit("/", 1)[-1]
    weights_path = get_file(filename, WEIGHT_PATH, cache_subdir="paz/models")
    print("Loading %s model weights" % weights_path)
    load_weights_by_name(model, weights_path)


# The legacy weights load by layer name, not by save order, because the clean
# model builds its branches in a different order than the original graph.
def load_weights_by_name(model, weights_path):
    with h5py.File(weights_path, "r") as weights:
        for layer_name in decode(weights.attrs["layer_names"]):
            group = weights[layer_name]
            weight_names = decode(group.attrs.get("weight_names", []))
            if weight_names:
                arrays = [np.array(group[name]) for name in weight_names]
                model.get_layer(layer_name).set_weights(arrays)


def decode(names):
    return [n.decode() if isinstance(n, bytes) else n for n in names]


def validate_weights(weights):
    if weights not in [None, "COCO"]:
        raise ValueError(f"Invalid weights: {weights}")
