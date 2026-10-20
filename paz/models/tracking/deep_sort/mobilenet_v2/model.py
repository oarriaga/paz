"""MobileNetV2 (width 1.0, 224 input) returning pooled 1280-d features.

Written from Sandler et al. 2018, "MobileNetV2: Inverted Residuals and
Linear Bottlenecks" (arXiv:1801.04381). The classifier is left out: the
model ends at the global mean pool, as used for DeepSORT appearance
features. Padding and normalization follow the PyTorch weights it loads.
"""
from keras import Input, Model
from keras.layers import Add, BatchNormalization, Conv2D, DepthwiseConv2D
from keras.layers import GlobalAveragePooling2D, ReLU, ZeroPadding2D

from paz.models.tracking.deep_sort.mobilenet_v2 import pretrained


def MobileNetV2(weights="pretrained", name="mobilenet_v2"):
    image = Input((224, 224, 3), name="image")
    x = build_head(build_blocks(build_stem(image)))
    features = GlobalAveragePooling2D(name="pool")(x)
    model = Model(image, features, name=name)
    load_weights(model, weights)
    return model


def load_weights(model, weights):
    if weights == "pretrained":
        weights = pretrained.download_weights()
    if weights is not None:
        model.load_weights(weights)


def build_stem(x):
    # symmetric padding of 1, as in the PyTorch weights; "same" pads only
    # the bottom and right at stride 2
    x = ZeroPadding2D(1, name="stem_pad")(x)
    x = Conv2D(32, 3, strides=2, use_bias=False, name="stem_conv")(x)
    return activate(normalize(x, "stem"), "stem")


def build_blocks(x):
    for block_arg, config in enumerate(list_block_configs()):
        x = build_inverted_residual(x, *config, f"block_{block_arg}")
    return x


def list_block_configs():
    stages = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2))
    stages = stages + ((6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))
    configs = []
    for expansion, out_channels, num_repeats, stride in stages:
        for block_stride in [stride] + [1] * (num_repeats - 1):
            configs.append((expansion, out_channels, block_stride))
    return configs


def build_inverted_residual(x, expansion, out_channels, stride, name):
    y = build_expansion(x, expansion, f"{name}_expand")
    y = build_depthwise(y, stride, f"{name}_depthwise")
    y = build_pointwise(y, out_channels, f"{name}_project")
    is_residual = stride == 1 and x.shape[-1] == out_channels
    return Add(name=f"{name}_add")([x, y]) if is_residual else y


def build_expansion(x, expansion, name):
    if expansion == 1:
        expanded = x
    else:
        num_filters = expansion * x.shape[-1]
        expanded = activate(build_pointwise(x, num_filters, name), name)
    return expanded


def build_depthwise(x, stride, name):
    x = ZeroPadding2D(1, name=f"{name}_pad")(x)
    kwargs = dict(strides=stride, use_bias=False, name=f"{name}_conv")
    x = DepthwiseConv2D(3, **kwargs)(x)
    return activate(normalize(x, name), name)


def build_pointwise(x, num_filters, name):
    x = Conv2D(num_filters, 1, use_bias=False, name=f"{name}_conv")(x)
    return normalize(x, name)


def build_head(x):
    return activate(build_pointwise(x, 1280, "head"), "head")


def normalize(x, name):
    # the PyTorch default epsilon; the Keras default 1e-3 shifts features
    return BatchNormalization(epsilon=1e-5, name=f"{name}_bn")(x)


def activate(x, name):
    return ReLU(max_value=6.0, name=f"{name}_relu")(x)
