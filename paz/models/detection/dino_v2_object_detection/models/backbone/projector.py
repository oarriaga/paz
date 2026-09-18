from functools import partial

from keras import Input, Model, ops, layers

LEAKY_RELU = partial(layers.LeakyReLU, negative_slope=0.1)
ACTIVATION_KEYS = ("silu", "relu", "LeakyReLU", "leakyrelu", "lrelu", None)
ACTIVATION_VALUES = (partial(layers.Activation, "silu"), partial(layers.Activation, "relu"), LEAKY_RELU, LEAKY_RELU, LEAKY_RELU, layers.Identity)  # fmt: skip
ACTIVATION_FACTORIES = dict(zip(ACTIVATION_KEYS, ACTIVATION_VALUES))


def get_norm(norm, name=None):
    result = norm
    if isinstance(norm, str) and len(norm) == 0:
        result = None
    elif norm == "LN":
        result = layers.LayerNormalization(epsilon=1e-6, name=name)
    return result


def get_activation(name):
    # Only the selected factory runs, so no unused layer is ever created.
    if name not in ACTIVATION_FACTORIES:
        raise AttributeError("Unsupported act type: {}".format(name))
    return ACTIVATION_FACTORIES[name]()


def MultiScaleProjector(in_channels, out_channels, scale_factors, input_scales=None, num_blocks=3, layer_norm=False, rms_norm=False, name="projector"):  # fmt: skip
    if input_scales is None:
        input_scales = scale_factors
    assert len(input_scales) == len(in_channels), "input_scales must match in_channels length"  # fmt: skip
    inputs = build_projector_inputs(in_channels, name)
    config = (in_channels, out_channels, scale_factors, input_scales,
              num_blocks, layer_norm, rms_norm)
    outputs = build_multiscale_projector(inputs, *config)
    return Model(inputs, outputs, name=name)


def build_projector_inputs(in_channels, name):
    inputs = []
    for index, channels in enumerate(in_channels):
        shape = (None, None, channels)
        inputs.append(Input(shape, name=f"{name}_input_{index}"))
    return inputs


def build_multiscale_projector(inputs, in_channels, out_channels, scale_factors, input_scales, num_blocks, layer_norm, rms_norm):  # fmt: skip
    results = []
    use_extra_pool = False
    for stage, target_scale in enumerate(scale_factors):
        sampled = sample_stage(inputs, in_channels, target_scale, input_scales, layer_norm, stage)  # fmt: skip
        if target_scale == 0.25:
            use_extra_pool = True
        else:
            c1 = combined_channels(in_channels, target_scale, input_scales)
            args = (fuse_features(sampled), c1, out_channels, num_blocks,
                    False, 1, 0.5, "silu", layer_norm, rms_norm, f"stage_{stage}_c2f")  # fmt: skip
            refined = build_c2f(*args)
            results.append(get_norm("LN", name=f"stage_{stage}_norm")(refined))  # fmt: skip
    if use_extra_pool:
        results.append(apply_extra_pool(results[-1]))
    return results


def sample_stage(inputs, in_channels, target_scale, input_scales, layer_norm, stage):  # fmt: skip
    sampled = []
    for index, input_scale in enumerate(input_scales):
        ratio = target_scale / input_scale
        name = f"stage_{stage}_samp_{index}"
        sampled.append(build_sampling(inputs[index], in_channels[index], ratio, layer_norm, name))  # fmt: skip
    return sampled


def build_sampling(x, in_dim, ratio, layer_norm, name):
    # The ratios are mutually exclusive, so exactly one branch builds layers
    # and the fall-through case builds none.
    result = x
    if ratio == 4.0:
        result = upsample_four(x, in_dim, name)
    if ratio == 2.0:
        result = build_conv_transpose(x, in_dim // 2, f"{name}_ctx1")
    if ratio == 0.5:
        args = (x, in_dim, in_dim, 3, 2, 1, 1, "relu", layer_norm, False)
        result = build_conv_x(*args, f"{name}_cvx")
    return result


def upsample_four(x, in_dim, name):
    x = build_conv_transpose(x, in_dim // 2, f"{name}_ctx1")
    x = get_norm("LN", name=f"{name}_ctx1_norm")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    return build_conv_transpose(x, in_dim // 4, f"{name}_ctx2")


def build_conv_transpose(x, filters, name):
    args = dict(kernel_size=2, strides=2, padding="valid", name=name)
    return layers.Conv2DTranspose(filters, **args)(x)


def fuse_features(sampled):
    return ops.concatenate(sampled, axis=-1) if len(sampled) > 1 else sampled[0]


def combined_channels(in_channels, target_scale, input_scales):
    total = 0
    for index, input_scale in enumerate(input_scales):
        ratio = target_scale / input_scale
        if ratio >= 1.0:
            total = total + in_channels[index] // int(ratio)
        else:
            total = total + in_channels[index]
    return total


def apply_extra_pool(x):
    return layers.MaxPooling2D(pool_size=1, strides=2, padding="valid")(x)


def SimpleProjector(in_dim, out_dim, factor_kernel=False, name="simple_projector"):  # fmt: skip
    x = Input((None, None, in_dim), name=f"{name}_input")
    output = build_simple_projector(x, in_dim, out_dim, factor_kernel)
    return Model([x], [output], name=name)


def build_simple_projector(x, in_dim, out_dim, factor_kernel):
    if factor_kernel:
        x = build_conv_x(x, in_dim, out_dim, (3, 1), 1, 1, 1, "silu", True, False, "convx1")  # fmt: skip
        x = build_conv_x(x, out_dim, out_dim, (1, 3), 1, 1, 1, "silu", True, False, "convx2")  # fmt: skip
    else:
        x = build_conv_x(x, in_dim, in_dim * 2, 3, 1, 1, 1, "silu", True, False, "convx1")  # fmt: skip
        x = build_conv_x(x, in_dim * 2, out_dim, 3, 1, 1, 1, "silu", True, False, "convx2")  # fmt: skip
    return get_norm("LN", name="ln")(x)


def build_c2f(x, c1, c2, n, shortcut, g, e, act, layer_norm, rms_norm, name):  # fmt: skip
    c = int(c2 * e)
    args = (x, c1, 2 * c, 1, 1, 1, 1, act, layer_norm, rms_norm, f"{name}_cv1")
    y = list(ops.split(build_conv_x(*args), 2, axis=-1))
    for index in range(n):
        args = (y[-1], c, c, shortcut, g, (3, 3), 1.0, act, layer_norm, rms_norm, f"{name}_m_{index}")  # fmt: skip
        y.append(build_bottleneck(*args))
    args = (ops.concatenate(y, axis=-1), (2 + n) * c, c2, 1, 1, 1, 1, act, layer_norm, rms_norm, f"{name}_cv2")  # fmt: skip
    return build_conv_x(*args)


def build_bottleneck(x, c1, c2, shortcut, g, k, e, act, layer_norm, rms_norm, name):  # fmt: skip
    c_ = int(c2 * e)
    args = (x, c1, c_, k[0], 1, 1, 1, act, layer_norm, rms_norm, f"{name}_cv1")
    branch = build_conv_x(*args)
    args = (branch, c_, c2, k[1], 1, g, 1, act, layer_norm, rms_norm, f"{name}_cv2")  # fmt: skip
    branch = build_conv_x(*args)
    return x + branch if (shortcut and c1 == c2) else branch


# in_planes is unused but stays in the signature: it mirrors the upstream
# PyTorch ConvX positional order that test_projector.py builds against.
def build_conv_x(x, in_planes, out_planes, kernel, stride, groups, dilation, act, layer_norm, rms_norm, name):  # fmt: skip
    kernel_size = (kernel, kernel) if isinstance(kernel, int) else kernel
    x = pad_for_kernel(x, kernel_size)
    x = build_conv(x, out_planes, kernel_size, stride, groups, dilation, name)
    x = normalize_conv(x, layer_norm, rms_norm, name)
    return get_activation(act)(x)


def pad_for_kernel(x, kernel_size):
    height = kernel_size[0] // 2
    width = kernel_size[1] // 2
    pad = ((height, height), (width, width))
    return layers.ZeroPadding2D(padding=pad)(x)


def build_conv(x, out_planes, kernel_size, stride, groups, dilation, name):
    keys = ("kernel_size", "strides", "padding", "groups", "dilation_rate",
            "use_bias", "name")
    values = (kernel_size, stride, "valid", groups, dilation, False,
              f"{name}_conv")
    return layers.Conv2D(out_planes, **dict(zip(keys, values)))(x)


def normalize_conv(x, layer_norm, rms_norm, name):
    if rms_norm:
        raise NotImplementedError("RMSNorm not implemented yet")
    if layer_norm:
        result = get_norm("LN", name=f"{name}_bn")(x)
    else:
        result = layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn")(x)
    return result
