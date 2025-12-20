import keras
from keras import layers
from keras import ops


@keras.saving.register_keras_serializable()
class Identity(keras.Layer):
    """Identity layer to match nn.Identity"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x


@keras.saving.register_keras_serializable()
class LayerNorm(keras.Layer):
    """
    Keras 3 implementation of ConvNeXt LayerNorm.

    This layer expects input in 'channels_first' (NCHW) format, transposes it to
    'channels_last' (NHWC) for normalization, and then transposes it back.
    This matches the specific behavior required by the Dino/ConvNeXt backbone projector.
    """

    def __init__(self, normalized_shape, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=self.normalized_shape,
            initializer="ones",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=self.normalized_shape,
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # Input is NCHW (channels_first).
        x = ops.transpose(x, (0, 2, 3, 1))

        # Calculate mean and variance across the channel dimension
        mean = ops.mean(x, axis=-1, keepdims=True)
        var = ops.var(x, axis=-1, keepdims=True)

        x = (x - mean) / ops.sqrt(var + self.eps)
        x = x * self.weight + self.bias

        # Transpose back to NCHW
        x = ops.transpose(x, (0, 3, 1, 2))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "normalized_shape": self.normalized_shape,
                "eps": self.eps,
            }
        )
        return config


def get_norm(norm, out_channels):
    if norm == "LN":
        return LayerNorm(out_channels, eps=1e-6)
    return Identity()


def get_activation(name):
    if name == "silu":
        return layers.Activation("silu")
    elif name == "relu":
        return layers.Activation("relu")
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        return layers.LeakyReLU(negative_slope=0.1)
    elif name == "gelu":
        # Explicitly use exact=True (approximate=False) to match PyTorch GELU
        return layers.Activation(lambda x: keras.activations.gelu(x, approximate=False))
    return Identity()


@keras.saving.register_keras_serializable()
class ConvX(keras.Layer):
    """
    A standard convolution block: Conv2D -> Norm -> Act.
    Implements manual padding to match typical PyTorch padding logic if needed.
    """

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel=3,
        stride=1,
        groups=1,
        dilation=1,
        act="relu",
        layer_norm=False,
        rms_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.act_name = act
        self.layer_norm = layer_norm
        self.rms_norm = rms_norm

        # Padding calculation
        pad_h = self.kernel[0] // 2
        pad_w = self.kernel[1] // 2
        self.pad_layer = layers.ZeroPadding2D(
            padding=((pad_h, pad_h), (pad_w, pad_w)), data_format="channels_first"
        )

        # Use valid padding because we manually padded above.
        self.conv = layers.Conv2D(
            filters=out_planes,
            kernel_size=self.kernel,
            strides=stride,
            padding="valid",
            groups=groups,
            dilation_rate=dilation,
            use_bias=False,
            data_format="channels_first",
        )

        if rms_norm:
            self.bn = layers.RMSNormalization(axis=1)
        elif layer_norm:
            self.bn = get_norm("LN", out_planes)
        else:
            self.bn = layers.BatchNormalization(axis=1, epsilon=1e-5)

        self.act = get_activation(act)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        x = self.pad_layer(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_planes": self.in_planes,
                "out_planes": self.out_planes,
                "kernel": self.kernel,
                "stride": self.stride,
                "groups": self.groups,
                "dilation": self.dilation,
                "act": self.act_name,
                "layer_norm": self.layer_norm,
                "rms_norm": self.rms_norm,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class Bottleneck(keras.Layer):
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
        act="silu",
        layer_norm=False,
        rms_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.shortcut = shortcut
        self.g = g
        self.k = k
        self.e = e
        self.act_name = act
        self.layer_norm = layer_norm
        self.rms_norm = rms_norm

        c_ = int(c2 * e)
        k_tuple = (k, k) if isinstance(k, int) else k

        self.cv1 = ConvX(
            c1, c_, k_tuple[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.cv2 = ConvX(
            c_,
            c2,
            k_tuple[1],
            1,
            groups=g,
            act=act,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )
        self.add = shortcut and (c1 == c2)

    def build(self, input_shape):
        self.cv1.build(input_shape)
        # Infer the output shape of cv1 to build cv2
        shape_cv1 = list(input_shape)
        if shape_cv1[1] is not None:
            shape_cv1[1] = self.cv1.out_planes
        self.cv2.build(tuple(shape_cv1))
        super().build(input_shape)

    def call(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "c1": self.c1,
                "c2": self.c2,
                "shortcut": self.shortcut,
                "g": self.g,
                "k": self.k,
                "e": self.e,
                "act": self.act_name,
                "layer_norm": self.layer_norm,
                "rms_norm": self.rms_norm,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class C2f(keras.Layer):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        act="silu",
        layer_norm=False,
        rms_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        self.g = g
        self.e = e
        self.act_name = act
        self.layer_norm = layer_norm
        self.rms_norm = rms_norm

        self.c = int(c2 * e)

        self.cv1 = ConvX(
            c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )
        self.bottlenecks = [
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                (3, 3),
                1.0,
                act,
                layer_norm,
                rms_norm,
                name=f"bottleneck_{i}",
            )
            for i in range(n)
        ]
        self.cv2 = ConvX(
            (2 + n) * self.c,
            c2,
            1,
            1,
            act=act,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

    def build(self, input_shape):
        self.cv1.build(input_shape)
        # CV1 splits channels into 2 chunks of size self.c
        bottleneck_shape = (input_shape[0], self.c, input_shape[2], input_shape[3])
        for b in self.bottlenecks:
            b.build(bottleneck_shape)

        # Concatenation of (split_1, split_2, b1, b2, ... bn) -> (2 + n) * c
        cv2_input_shape = (
            input_shape[0],
            (2 + self.n) * self.c,
            input_shape[2],
            input_shape[3],
        )
        self.cv2.build(cv2_input_shape)
        super().build(input_shape)

    def call(self, x):
        x_cv1 = self.cv1(x)
        y = list(ops.split(x_cv1, indices_or_sections=2, axis=1))

        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))

        return self.cv2(ops.concatenate(y, axis=1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "c1": self.c1,
                "c2": self.c2,
                "n": self.n,
                "shortcut": self.shortcut,
                "g": self.g,
                "e": self.e,
                "act": self.act_name,
                "layer_norm": self.layer_norm,
                "rms_norm": self.rms_norm,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class SimpleProjector(keras.Layer):
    def __init__(self, in_dim, out_dim, factor_kernel=False, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor_kernel = factor_kernel

        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim * 2, 3, layer_norm=True, act="silu")
            self.convx2 = ConvX(in_dim * 2, out_dim, 3, layer_norm=True, act="silu")
        else:
            self.convx1 = ConvX(in_dim, out_dim, (3, 1), layer_norm=True, act="silu")
            self.convx2 = ConvX(out_dim, out_dim, (1, 3), layer_norm=True, act="silu")

        self.ln = get_norm("LN", out_dim)

    def build(self, input_shape):
        feat_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        self.convx1.build(feat_shape)

        c1_out = self.out_dim if self.factor_kernel else self.in_dim * 2
        # Intermediate shape assuming spatial dims are preserved
        inter_shape = (feat_shape[0], c1_out, feat_shape[2], feat_shape[3])

        self.convx2.build(inter_shape)
        # Explicit build for LN to ensure weights are created before copy
        self.ln.build((feat_shape[0], self.out_dim, feat_shape[2], feat_shape[3]))
        super().build(input_shape)

    def call(self, x):
        # x is a list of features, projector uses the first one
        x_in = x[0]
        x_out = self.convx2(self.convx1(x_in))
        return [self.ln(x_out)]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "factor_kernel": self.factor_kernel,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class MultiScaleProjector(keras.Layer):
    """
    Keras implementation of MultiScaleProjector.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels_list = in_channels
        self.out_channels = out_channels
        self.scale_factors = scale_factors
        self.num_blocks = num_blocks
        self.layer_norm = layer_norm
        self.rms_norm = rms_norm
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        self.use_extra_pool = False
        self.stages_sampling = []
        self.stages = []
        self.stage_in_dims_agg = []

        for scale in scale_factors:
            # Special case for scale 0.25 (P6/P7 pooling usually)
            if scale == 0.25:
                self.use_extra_pool = True
                continue

            current_stage_sampling = []

            # 1. Sampler Stages (Up/Down-sampling)
            for in_dim in self.in_channels_list:
                layers_list = []

                if scale == 4.0:
                    layers_list.extend(
                        [
                            layers.Conv2DTranspose(
                                filters=in_dim // 2,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                use_bias=True,
                                data_format="channels_first",
                            ),
                            get_norm("LN", in_dim // 2),
                            get_activation("gelu"),
                            layers.Conv2DTranspose(
                                filters=in_dim // 4,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                use_bias=True,
                                data_format="channels_first",
                            ),
                        ]
                    )
                elif scale == 2.0:
                    layers_list.extend(
                        [
                            layers.Conv2DTranspose(
                                filters=in_dim // 2,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                use_bias=True,
                                data_format="channels_first",
                            )
                        ]
                    )
                elif scale == 1.0:
                    layers_list.append(Identity())
                elif scale == 0.5:
                    layers_list.append(
                        ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm)
                    )
                else:
                    raise NotImplementedError(f"Unsupported scale: {scale}")

                current_stage_sampling.append(keras.Sequential(layers_list))

            self.stages_sampling.append(current_stage_sampling)

            # 2. Fusion Stage (C2f + Norm)
            # Determine aggregated input dimension
            in_dim_agg = int(sum(ch // max(1, scale) for ch in self.in_channels_list))
            self.stage_in_dims_agg.append(in_dim_agg)

            self.stages.append(
                keras.Sequential(
                    [
                        C2f(
                            in_dim_agg, out_channels, num_blocks, layer_norm=layer_norm
                        ),
                        get_norm("LN", out_channels),
                    ]
                )
            )

        # 3. Extra Pool Layer
        if self.use_extra_pool:
            self.extra_pool_layer = layers.MaxPooling2D(
                pool_size=1, strides=2, padding="valid", data_format="channels_first"
            )

    def build(self, input_shape):
        # 1. Build Samplers
        for i, stage_group in enumerate(self.stages_sampling):
            for j, sampler in enumerate(stage_group):
                sampler.build(input_shape[j])

        # 2. Build Fusion Stages
        for i, stage in enumerate(self.stages):
            first_sampler = self.stages_sampling[i][0]
            first_out_shape = first_sampler.compute_output_shape(input_shape[0])

            in_dim_agg = self.stage_in_dims_agg[i]
            stage_input_shape = (
                first_out_shape[0],
                in_dim_agg,
                first_out_shape[2],
                first_out_shape[3],
            )
            stage.build(stage_input_shape)

        # 3. Build Extra Pool Layer
        if self.use_extra_pool and len(self.stages) > 0:
            last_stage = self.stages[-1]
            i = len(self.stages) - 1
            first_sampler = self.stages_sampling[i][0]
            first_out_shape = first_sampler.compute_output_shape(input_shape[0])

            in_dim_agg = self.stage_in_dims_agg[i]
            stage_input_shape = (
                first_out_shape[0],
                in_dim_agg,
                first_out_shape[2],
                first_out_shape[3],
            )

            last_result_shape = last_stage.compute_output_shape(stage_input_shape)
            self.extra_pool_layer.build(last_result_shape)

        super().build(input_shape)

    def call(self, x, training=False):
        x = list(x)
        num_features = len(x)

        if training and self.survival_prob < 1.0:
            final_drop_prob = 1.0 - self.survival_prob
            drop_p = ops.cast(ops.random.uniform((), 0.0, 1.0), dtype=x[0].dtype)

            for i in range(1, num_features):
                critical_drop_prob = float(i * (final_drop_prob / (num_features - 1)))
                critical_drop_prob_tensor = ops.cast(
                    critical_drop_prob, dtype=x[0].dtype
                )

                # Zero out specific feature map if dropped
                x[i] = ops.where(
                    ops.less(drop_p, critical_drop_prob_tensor),
                    ops.zeros_like(x[i]),
                    x[i],
                )

        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                x[-(i + 1)] = ops.zeros_like(x[-(i + 1)])

        results = []
        for i, stage in enumerate(self.stages):
            # Apply sampling for this pyramid level
            feat_fuse = [
                sampler(x[j]) for j, sampler in enumerate(self.stages_sampling[i])
            ]

            # Fuse features
            if len(feat_fuse) > 1:
                feat_fuse_out = ops.concatenate(feat_fuse, axis=1)
            else:
                feat_fuse_out = feat_fuse[0]

            # Apply C2f stage
            results.append(stage(feat_fuse_out))

        if self.use_extra_pool and len(results) > 0:
            results.append(self.extra_pool_layer(results[-1]))

        return results

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels_list,
                "out_channels": self.out_channels,
                "scale_factors": self.scale_factors,
                "num_blocks": self.num_blocks,
                "layer_norm": self.layer_norm,
                "rms_norm": self.rms_norm,
                "survival_prob": self.survival_prob,
                "force_drop_last_n_features": self.force_drop_last_n_features,
            }
        )
        return config
