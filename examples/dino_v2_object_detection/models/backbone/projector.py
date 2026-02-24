import keras
from keras import ops
from keras import layers
import numpy as np


@keras.saving.register_keras_serializable(package="RFDETR")
class LayerNorm(layers.Layer):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(input_shape[-1],), initializer="ones", name="weight", trainable=True
        )
        self.bias = self.add_weight(
            shape=(input_shape[-1],), initializer="zeros", name="bias", trainable=True
        )

    def call(self, x):
        # Expecting NHWC in Keras usually, but this docstring says NCHW.
        # However, standard Keras LayerNorm works on the last dimension by default.
        # If the input is NHWC, we normalize over C.
        u = ops.mean(x, axis=-1, keepdims=True)
        s = ops.mean(ops.square(x - u), axis=-1, keepdims=True)
        x = (x - u) / ops.sqrt(s + self.epsilon)
        return self.weight * x + self.bias


def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        if norm == "LN":
            # Keras LayerNormalization defaults to axis=-1 which is correct for NHWC
            return layers.LayerNormalization(epsilon=1e-6)
    return norm


def get_activation(name):
    if name == "silu":
        return layers.Activation("silu")
    elif name == "relu":
        return layers.Activation("relu")
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        return layers.LeakyReLU(negative_slope=0.1)
    elif name is None:
        return layers.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))


@keras.saving.register_keras_serializable(package="RFDETR")
class ConvX(layers.Layer):
    """Conv-bn module"""

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
        self._in_planes = in_planes
        self._out_planes = out_planes
        self._kernel = kernel
        self._stride = stride
        self._groups = groups
        self._dilation = dilation
        self._act = act
        self._layer_norm = layer_norm
        self._rms_norm = rms_norm
        if isinstance(kernel, int):
            kernel_size = (kernel, kernel)
        else:
            kernel_size = kernel

        # Match PyTorch padding: (kernel // 2)
        # PyTorch uses symmetric padding for both height and width
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        self.pad = layers.ZeroPadding2D(padding=((pad_h, pad_h), (pad_w, pad_w)))

        self.conv = layers.Conv2D(
            out_planes,
            kernel_size=kernel_size,
            strides=stride,
            padding="valid",  # We handle padding manually
            groups=groups,
            dilation_rate=dilation,
            use_bias=False,
            name="conv",
        )

        if rms_norm:
            raise NotImplementedError("RMSNorm not implemented yet")
        else:
            if layer_norm:
                self.bn = get_norm("LN", out_planes)
            else:
                self.bn = layers.BatchNormalization(name="bn", epsilon=1e-5)

        self.act = get_activation(act)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_planes": self._in_planes,
                "out_planes": self._out_planes,
                "kernel": self._kernel,
                "stride": self._stride,
                "groups": self._groups,
                "dilation": self._dilation,
                "act": self._act,
                "layer_norm": self._layer_norm,
                "rms_norm": self._rms_norm,
            }
        )
        return config

    def call(self, x):
        return self.act(self.bn(self.conv(self.pad(x))))


@keras.saving.register_keras_serializable(package="RFDETR")
class Bottleneck(layers.Layer):
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
        self._c1 = c1
        self._c2 = c2
        self._shortcut = shortcut
        self._g = g
        self._k = k
        self._e = e
        self._act = act
        self._layer_norm = layer_norm
        self._rms_norm = rms_norm
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(
            c1,
            c_,
            k[0],
            1,
            act=act,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            name="cv1",
        )
        self.cv2 = ConvX(
            c_,
            c2,
            k[1],
            1,
            groups=g,
            act=act,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            name="cv2",
        )
        self.add = shortcut and c1 == c2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "c1": self._c1,
                "c2": self._c2,
                "shortcut": self._shortcut,
                "g": self._g,
                "k": self._k,
                "e": self._e,
                "act": self._act,
                "layer_norm": self._layer_norm,
                "rms_norm": self._rms_norm,
            }
        )
        return config

    def call(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@keras.saving.register_keras_serializable(package="RFDETR")
class C2f(layers.Layer):
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
        self._c1 = c1
        self._c2 = c2
        self._n = n
        self._shortcut = shortcut
        self._g = g
        self._e = e
        self._act = act
        self._layer_norm = layer_norm
        self._rms_norm = rms_norm
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(
            c1,
            2 * self.c,
            1,
            1,
            act=act,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            name="cv1",
        )
        self.cv2 = ConvX(
            (2 + n) * self.c,
            c2,
            1,
            act=act,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            name="cv2",
        )
        self.m = [
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=(3, 3),
                e=1.0,
                act=act,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
                name=f"m_{i}",
            )
            for i in range(n)
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "c1": self._c1,
                "c2": self._c2,
                "n": self._n,
                "shortcut": self._shortcut,
                "g": self._g,
                "e": self._e,
                "act": self._act,
                "layer_norm": self._layer_norm,
                "rms_norm": self._rms_norm,
            }
        )
        return config

    def call(self, x):
        # cv1 output shape: (B, H, W, 2*c)
        y = self.cv1(x)
        # split along channel axis (last axis)
        y = ops.split(y, 2, axis=-1)
        y = list(y)  # make it a list
        # y is now [y0, y1] each (B, H, W, c)

        # Apply bottlenecks to the last chunk (originally y[1] aka y[-1])
        # In PyTorch: y.extend(m(y[-1]) for m in self.m)
        # It creates a chain of outputs? No, it appends new outputs to y list.
        # Wait, the loop `for m in self.m: m(y[-1])` takes the *last* element, processes it, and adds it.
        # So it IS a chain if it always takes the latest.
        # But `m(y[-1])` suggests it takes the *current* last element.
        # And then appends the result. So the *next* m takes the result of the *previous* m.
        # Yes, that's correct.

        for m in self.m:
            y.append(m(y[-1]))

        # Concatenate all parts
        return self.cv2(ops.concatenate(y, axis=-1))


@keras.saving.register_keras_serializable(package="RFDETR")
class MultiScaleProjector(layers.Layer):
    """
    MultiScaleProjector
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        input_scales=None,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_blocks = num_blocks
        self._layer_norm = layer_norm
        self._rms_norm = rms_norm
        self.scale_factors = scale_factors
        if input_scales is None:
            self.input_scales = scale_factors
        else:
            self.input_scales = input_scales
        assert len(self.input_scales) == len(
            in_channels
        ), "input_scales must match in_channels length"

        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features
        self.use_extra_pool = False

        self.stages_sampling_blocks = []  # List of lists of layers
        self.stages_blocks = []  # List of layers

        # We assume input 'x' is a list of features corresponding to scale_factors
        # e.g. x[0] -> scale_factors[0]
        # So we iterate over target scales (i) and input scales (j)

        for i, target_scale in enumerate(scale_factors):
            current_stage_sampling = []

            # For each input feature map 'j', transform it to 'target_scale'
            for j, input_scale in enumerate(self.input_scales):
                in_dim = in_channels[j]

                # Calculate ratio: target / input
                # e.g. Target P3 (scale 2.0), Input P4 (scale 1.0) -> ratio 2.0 (Upsample x2)
                # e.g. Target P3 (scale 2.0), Input P3 (scale 2.0) -> ratio 1.0 (Identity)
                # e.g. Target P4 (scale 1.0), Input P3 (scale 2.0) -> ratio 0.5 (Downsample x2)

                ratio = target_scale / input_scale

                layers_list = []

                if ratio == 4.0:
                    # Upsample x4: ConvT (x2) -> LN -> GELU -> ConvT (x2)
                    # MATCH PYTORCH: channels are reduced (in_dim // 2 -> in_dim // 4)
                    layers_list.extend(
                        [
                            layers.Conv2DTranspose(
                                in_dim // 2,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                name=f"stage_{i}_samp_{j}_ctx1",
                            ),
                            get_norm("LN", in_dim // 2),
                            layers.Activation("gelu"),
                            layers.Conv2DTranspose(
                                in_dim // 4,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                name=f"stage_{i}_samp_{j}_ctx2",
                            ),
                        ]
                    )

                elif ratio == 2.0:
                    # Upsample x2: ConvT (x2)
                    # MATCH PYTORCH: channels are reduced (in_dim // 2)
                    layers_list.extend(
                        [
                            layers.Conv2DTranspose(
                                in_dim // 2,
                                kernel_size=2,
                                strides=2,
                                padding="valid",
                                name=f"stage_{i}_samp_{j}_ctx1",
                            ),
                        ]
                    )

                elif ratio == 1.0:
                    # Identity
                    pass

                elif ratio == 0.5:
                    # Downsample x2: ConvX stride 2
                    layers_list.extend(
                        [
                            ConvX(
                                in_dim,
                                in_dim,
                                3,
                                2,
                                layer_norm=layer_norm,
                                name=f"stage_{i}_samp_{j}_cvx",
                            ),
                        ]
                    )

                elif ratio == 0.25:
                    # Downsample x4
                    # Matches PyTorch use_extra_pool logic (skipped here)
                    # But if we enter here (e.g. target_scale 0.25 vs input 1.0), we should downsample?
                    # PyTorch implementation handles scale 0.25 by setting use_extra_pool=True and CONTINUING loop.
                    # It NEVER builds a sampler for 0.25.
                    # So we should strictly follow that structure if target_scale == 0.25.
                    pass

                elif ratio == 0.125:  # Upsample 0.25 -> 2.0? ratio 8.0?
                    # Unsupported
                    pass

                else:
                    # Fallback or error?
                    pass

                if layers_list:
                    current_stage_sampling.append(
                        keras.Sequential(layers_list, name=f"stage_{i}_samp_{j}")
                    )
                else:
                    current_stage_sampling.append(
                        layers.Identity(name=f"stage_{i}_samp_{j}")
                    )

            # If we skipped due to extra pool (scale 0.25), continue
            if target_scale == 0.25:
                self.use_extra_pool = True
                continue

            self.stages_sampling_blocks.append(current_stage_sampling)

            # Combined input dim calculation matching PyTorch logic
            # PyTorch: int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            # Here 'scale' refers to the ratio (target/input) essentially, if input is 1.0.
            # But technically it's the reduction factor applied.
            # If ratio 4.0 (Upsample), channels became in // 4.
            # If ratio 2.0 (Upsample), channels became in // 2.
            # If ratio 1.0, channels in.
            # If ratio 0.5 (Downsample), channels in.
            # So divisor is ratio if ratio >= 1 else 1.

            # We iterate over inputs again to calculate sum
            in_dim_combined = 0
            for j, input_scale in enumerate(self.input_scales):
                ratio = target_scale / input_scale
                if ratio >= 1.0:
                    divisor = int(ratio)
                    in_dim_combined += in_channels[j] // divisor
                else:
                    in_dim_combined += in_channels[j]

            layers_stage = [
                C2f(
                    in_dim_combined,
                    out_channels,
                    num_blocks,
                    layer_norm=layer_norm,
                    name=f"stage_{i}_c2f",
                ),
                get_norm("LN", out_channels),
            ]
            self.stages_blocks.append(
                keras.Sequential(layers_stage, name=f"stage_{i}_seq")
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self._in_channels,
                "out_channels": self._out_channels,
                "scale_factors": self.scale_factors,
                "input_scales": self.input_scales,
                "num_blocks": self._num_blocks,
                "layer_norm": self._layer_norm,
                "rms_norm": self._rms_norm,
                "survival_prob": self.survival_prob,
                "force_drop_last_n_features": self.force_drop_last_n_features,
            }
        )
        return config

    def call(self, x, training=False):
        # x is a list of feature maps (NHWC)

        # Handle Survival Prob / Drop features (Training only)
        if training and self.survival_prob < 1.0:
            pass  # TODO: Implement

        # Dropping last n features (always if set)
        if self.force_drop_last_n_features > 0:
            pass

        results = []

        for i, stage_model in enumerate(self.stages_blocks):
            feat_fuse = []
            sampling_modules = self.stages_sampling_blocks[i]

            for j, sampler in enumerate(sampling_modules):
                # x[j] is the j-th input feature map
                out = sampler(x[j])
                feat_fuse.append(out)

            # Concatenate all sampled inputs
            if len(feat_fuse) > 1:
                feat_fuse = ops.concatenate(feat_fuse, axis=-1)
            else:
                feat_fuse = feat_fuse[0]

            results.append(stage_model(feat_fuse))

        # Handle extra pool (P6)
        # Matches PyTorch: results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        if self.use_extra_pool:
            # Keras MaxPool: pool_size=1, strides=2, padding="valid" (0 padding)
            results.append(
                layers.MaxPooling2D(pool_size=1, strides=2, padding="valid")(
                    results[-1]
                )
            )

        return results


@keras.saving.register_keras_serializable(package="RFDETR")
class SimpleProjector(layers.Layer):
    def __init__(self, in_dim, out_dim, factor_kernel=False, **kwargs):
        super().__init__(**kwargs)
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._factor_kernel = factor_kernel
        if not factor_kernel:
            self.convx1 = ConvX(
                in_dim, in_dim * 2, layer_norm=True, act="silu", name="convx1"
            )
            self.convx2 = ConvX(
                in_dim * 2, out_dim, layer_norm=True, act="silu", name="convx2"
            )
        else:
            self.convx1 = ConvX(
                in_dim,
                out_dim,
                kernel=(3, 1),
                layer_norm=True,
                act="silu",
                name="convx1",
            )
            self.convx2 = ConvX(
                out_dim,
                out_dim,
                kernel=(1, 3),
                layer_norm=True,
                act="silu",
                name="convx2",
            )
        self.ln = get_norm("LN", out_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self._in_dim,
                "out_dim": self._out_dim,
                "factor_kernel": self._factor_kernel,
            }
        )
        return config

    def call(self, x):
        # x is expected to be a list, based on PyTorch forward: "x[0]"
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]
