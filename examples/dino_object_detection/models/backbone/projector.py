import math
import numpy as np
import keras
from keras import layers
from keras import ops
from keras import random


@keras.saving.register_keras_serializable()
class Identity(keras.Layer):
    """Identity layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x


@keras.saving.register_keras_serializable()
class LayerNorm(layers.Layer):
    """
    A LayerNorm variant that matches the PyTorch implementation exactly.
    Attributes are named 'weight' and 'bias' to match PyTorch-style access patterns
    in the test scripts.
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
        x = ops.transpose(x, (0, 2, 3, 1))

        # Manual LayerNorm
        mean = ops.mean(x, axis=-1, keepdims=True)
        var = ops.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * ops.rsqrt(var + self.eps)

        # Apply affine parameters
        x_out = x_norm * self.weight + self.bias

        # Transpose back to (N, C, H, W)
        return ops.transpose(x_out, (0, 3, 1, 2))

    def get_config(self):
        config = super().get_config()
        config.update({"normalized_shape": self.normalized_shape, "eps": self.eps})
        return config


def get_norm(norm, out_channels):
    """Factory function to return the correct Normalization layer."""
    if norm == "LN":
        return LayerNorm(out_channels, eps=1e-6)
    return Identity()


def get_activation(name):
    """Matches PyTorch activation getters."""
    if name == "silu":
        return layers.Activation("silu")
    elif name == "relu":
        return layers.Activation("relu")
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        return layers.LeakyReLU(negative_slope=0.1)
    elif name == "gelu":
        return layers.Activation("gelu")
    elif name is None:
        return Identity()
    return Identity()


@keras.saving.register_keras_serializable()
class ConvX(keras.Layer):
    """
    Conv-bn module.
    Replicates PyTorch padding logic (kernel // 2) exactly, which does
    not auto-compensate for dilation > 1.
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

        pad_h = self.kernel[0] // 2
        pad_w = self.kernel[1] // 2

        self.pad_layer = layers.ZeroPadding2D(
            padding=((pad_h, pad_h), (pad_w, pad_w)), data_format="channels_first"
        )

        self.conv = layers.Conv2D(
            filters=out_planes,
            kernel_size=self.kernel,
            strides=stride,
            padding="valid",
            groups=groups,
            dilation_rate=dilation,
            use_bias=False,
            data_format="channels_first",
            kernel_initializer="he_uniform",
        )

        if rms_norm:
            self.bn = layers.RMSNormalization(axis=1)
        elif layer_norm:
            self.bn = get_norm("LN", out_planes)
        else:
            self.bn = layers.BatchNormalization(axis=1, epsilon=1e-5, momentum=0.9)

        self.act = get_activation(act)

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
    """Standard bottleneck."""

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

        c_ = int(c2 * e)  # hidden channels
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

    def call(self, x):
        y = self.cv2(self.cv1(x))
        return (x + y) if self.add else y

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
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

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
class MultiScaleProjector(keras.Layer):
    """
    MultiScaleProjector.
    Creates pyramid features built on top of the input feature map.
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

        for idx, scale in enumerate(scale_factors):
            if scale == 0.25:
                self.use_extra_pool = True
                continue

            current_stage_sampling = []

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
                                kernel_initializer="he_uniform",
                            ),
                            get_norm("LN", in_dim // 2),
                            layers.Activation("gelu"),
                            layers.Conv2DTranspose(
                                filters=in_dim // 4,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                use_bias=True,
                                data_format="channels_first",
                                kernel_initializer="he_uniform",
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
                                kernel_initializer="he_uniform",
                            ),
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

            # Logic: sum(in_channel // max(1, scale))
            in_dim_agg = int(sum(ch // max(1, scale) for ch in self.in_channels_list))

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

        if self.use_extra_pool:
            self.extra_pool_layer = layers.MaxPooling2D(
                pool_size=1, strides=2, padding="valid", data_format="channels_first"
            )

    def build(self, input_shape):
        # Explicit build for Sequential layers within lists
        for i, stage_group in enumerate(self.stages_sampling):
            for j, sampler in enumerate(stage_group):
                sampler.build(input_shape[j])

        # Build Aggregation Stages (C2f)
        for i, stage in enumerate(self.stages):
            # Infer spatial shape from first sampler path
            first_sampler = self.stages_sampling[i][0]
            sample_out_shape = first_sampler.compute_output_shape(input_shape[0])

            # Calculate total channels from all sampler paths
            total_channels = 0
            for j, sampler in enumerate(self.stages_sampling[i]):
                s_out = sampler.compute_output_shape(input_shape[j])
                total_channels += s_out[1]

            stage_input_shape = (
                sample_out_shape[0],
                total_channels,
                sample_out_shape[2],
                sample_out_shape[3],
            )
            stage.build(stage_input_shape)

        super().build(input_shape)

    def call(self, x, training=False):
        # x is a list of tensors
        x_list = list(x)
        num_features = len(x_list)

        # Stochastic Drop of Input Features
        if training and self.survival_prob < 1.0:
            final_drop_prob = 1.0 - self.survival_prob
            drop_p = random.uniform((), minval=0.0, maxval=1.0)

            for i in range(1, num_features):
                critical_drop_prob = float(i * (final_drop_prob / (num_features - 1)))
                should_drop = ops.less(drop_p, critical_drop_prob)

                # Functional update
                x_list[i] = ops.where(should_drop, ops.zeros_like(x_list[i]), x_list[i])

        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                idx = -(i + 1)
                x_list[idx] = ops.zeros_like(x_list[idx])

        results = []

        for i, stage in enumerate(self.stages):
            # Sampling
            feat_fuse = [
                sampler(x_list[j]) for j, sampler in enumerate(self.stages_sampling[i])
            ]

            # Fuse
            if len(feat_fuse) > 1:
                feat_fuse_out = ops.concatenate(feat_fuse, axis=1)
            else:
                feat_fuse_out = feat_fuse[0]

            # C2f + Norm
            results.append(stage(feat_fuse_out))

        # Extra Pool
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


@keras.saving.register_keras_serializable()
class SimpleProjector(keras.Layer):
    """Simple Projector."""

    def __init__(self, in_dim, out_dim, factor_kernel=False, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor_kernel = factor_kernel

        if not factor_kernel:
            self.convx1 = ConvX(
                in_dim, in_dim * 2, kernel=3, layer_norm=True, act="silu"
            )
            self.convx2 = ConvX(
                in_dim * 2, out_dim, kernel=3, layer_norm=True, act="silu"
            )
        else:
            self.convx1 = ConvX(
                in_dim, out_dim, kernel=(3, 1), layer_norm=True, act="silu"
            )
            self.convx2 = ConvX(
                out_dim, out_dim, kernel=(1, 3), layer_norm=True, act="silu"
            )

        self.ln = get_norm("LN", out_dim)

    def call(self, x):
        x_in = x[0]
        x_out = self.convx2(self.convx1(x_in))
        out = self.ln(x_out)
        return [out]

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
