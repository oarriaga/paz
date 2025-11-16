import keras
import logging
from keras import ops
from functools import partial
from typing import List, Dict, Optional, Union, Sequence

logger = logging.getLogger("dinov3")
Tensor = keras.KerasTensor


def drop_path(x, drop_prob, training):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
    random_tensor = keep_prob + keras.random.uniform(shape, dtype=x.dtype)
    random_tensor = ops.floor(random_tensor)
    output = ops.divide(x, keep_prob) * random_tensor
    return output


class DropPath(keras.layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class LayerNorm(keras.layers.Layer):
    def __init__(
        self, normalized_shape, eps=1e-6, data_format="channels_last", **kwargs
    ):
        super().__init__(**kwargs)
        self.weight = self.add_weight(
            shape=normalized_shape,
            initializer="ones",
            trainable=True,
            name="weight",
        )
        self.bias = self.add_weight(
            shape=normalized_shape,
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        self.eps = eps

        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                f"data_format should be 'channels_last' or 'channels_first', got {self.data_format}"
            )

    def call(self, x):
        if self.data_format == "channels_last":
            mean = ops.mean(x, axis=-1, keepdims=True)
            variance = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
            x_norm = (x - mean) / ops.sqrt(variance + self.eps)
            return x_norm * self.weight + self.bias
        else:
            mean = ops.mean(x, axis=1, keepdims=True)
            variance = ops.mean(ops.square(x - mean), axis=1, keepdims=True)
            x_norm = (x - mean) / ops.sqrt(variance + self.eps)
            return x_norm * self.weight[:, None, None] + self.bias[:, None, None]


class Block(keras.layers.Layer):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=7,
            padding="same",
            depth_multiplier=1,
            name="dwconv",
            data_format="channels_first",
        )
        self.norm = LayerNorm(normalized_shape=(dim,), eps=1e-6)
        self.pwconv1 = keras.layers.Dense(4 * dim)
        self.act = keras.layers.Activation("gelu")
        self.pwconv2 = keras.layers.Dense(dim)
        self.gamma = (
            self.add_weight(
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
                name="gamma",
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            DropPath(drop_prob=drop_path)
            if drop_path > 0.0
            else keras.layers.Identity()
        )

    def call(self, x, training=None):
        input = x
        x = self.dwconv(x)
        x = ops.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.transpose(x, (0, 3, 1, 2))

        x = self.drop_path(x, training=training)

        x = input + x
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.pwconv2.units,
                "drop_path": (
                    self.drop_path.drop_prob
                    if hasattr(self.drop_path, "drop_prob")
                    else 0.0
                ),
                "layer_scale_init_value": (
                    self.gamma.numpy().mean() if self.gamma is not None else 0.0
                ),
            }
        )
        return config


class ConvNeXt(keras.Model):
    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: int | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            print(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs
        self.downsample_layers = []
        stem = keras.Sequential(
            [
                keras.layers.Conv2D(
                    dims[0],
                    kernel_size=4,
                    strides=4,
                    data_format="channels_first",
                ),
                LayerNorm(normalized_shape=(dims[0],), data_format="channels_first"),
            ]
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = keras.Sequential(
                [
                    LayerNorm(
                        normalized_shape=(dims[i],), data_format="channels_first"
                    ),
                    keras.layers.Conv2D(
                        dims[i + 1],
                        kernel_size=2,
                        strides=2,
                        data_format="channels_first",
                    ),
                ]
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = []
        db_rates = [x.item() for x in ops.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = keras.Sequential(
                [
                    Block(
                        dim=dims[i],
                        drop_path=db_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)

        self.head = keras.layers.Identity()
        self.embed_dim = dims[-1]
        self.embed_dims = dims
        self.n_blocks = len(self.downsample_layers)
        self.chunked_blocks = False
        self.n_storage_tokens = 0

        self.norms = [keras.layers.Identity() for i in range(3)]
        self.norms.append(self.norm)

        self.patch_size = patch_size  # for DINO compatibility
        self.input_pad_size = 4

    def init_weights(self):
        self.walk(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, keras.layers.LayerNormalization):
            if hasattr(layer, "gamma"):
                layer.gamma.assign(ops.ones_like(layer.gamma))
            if hasattr(layer, "beta"):
                layer.beta.assign(ops.zeros_like(layer.beta))

        if isinstance(layer, LayerNorm):  # Check for your custom class
            if hasattr(layer, "weight"):
                layer.weight.assign(ops.ones_like(layer.weight))
            if hasattr(layer, "bias"):
                layer.bias.assign(ops.zeros_like(layer.bias))

        if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
            initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
            new_kernel_value = initializer(shape=layer.kernel.shape)
            layer.kernel.assign(new_kernel_value)

            initializer = keras.initializers.Zeros()
            new_bias_value = initializer(shape=layer.bias.shape)
            layer.bias.assign(new_bias_value)

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        if not isinstance(x, list):
            return self.forward_features_list([x], [masks])[0]

        else:
            return self.forward_features_list(x, masks)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Tensor]
    ) -> List[Dict[str, Tensor]]:

        output = []
        for x, masks in zip(x_list, masks_list):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

            x_pool = keras.ops.mean(x, axis=[-2, -1])
            shape = keras.ops.shape(x)
            x_flat = keras.ops.reshape(x, (shape[0], shape[1], -1))
            x = keras.ops.transpose(x_flat, (0, 2, 1))
            x_pool_expanded = keras.ops.expand_dims(x_pool, axis=1)

            x_concat = keras.ops.concatenate([x_pool_expanded, x], axis=1)
            x_norm = self.norm(x_concat)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output

    def call(self, *args, training=None, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def _get_intermediate_layers(self, x, n=1):
        h, w = x.shape[-2:]
        output, total_block_len = [], len(self.downsample_layers)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = ops.mean(x, axis=[-2, -1])
                x_patches = x

                if self.patch_size is not None:
                    x_patches = keras.ops.image.resize(
                        x,
                        size=(h // self.patch_size, w // self.patch_size),
                        interpolation="bilinear",
                        antialias=True,
                        data_format="channels_first",
                    )

                output.append(
                    [
                        x_pool,
                        x_patches,
                    ]
                )
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            nchw_shapes = [ops.shape(out[-1]) for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]

            new_outputs = []
            for (cls_token, patches), norm_layer in zip(outputs, norms):
                shape = ops.shape(patches)
                patches_flat = ops.reshape(patches, (shape[0], shape[1], -1))
                patches_permuted = ops.transpose(patches_flat, (0, 2, 1))

                new_outputs.append(
                    (
                        norm_layer(cls_token),
                        norm_layer(patches_permuted),
                    )
                )
            outputs = new_outputs

            if reshape:
                outputs = [
                    (cls_token, ops.reshape(ops.transpose(patches, (0, 2, 1)), nchw))
                    for (cls_token, patches), nchw in zip(outputs, nchw_shapes)
                ]

        elif not reshape:
            new_outputs = []
            for cls_token, patches in outputs:
                shape = ops.shape(patches)
                patches_flat = ops.reshape(patches, (shape[0], shape[1], -1))
                patches_permuted = ops.transpose(patches_flat, (0, 2, 1))
                new_outputs.append((cls_token, patches_permuted))
            outputs = new_outputs

        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )
