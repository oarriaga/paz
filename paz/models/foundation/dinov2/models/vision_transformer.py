from functools import partial
import math

from keras import Model, layers, ops, initializers
import numpy as np

from paz.models.foundation.dinov2.layers import (
    MLP,
    PatchEmbed,
    SwiGLUFFNFused,
    Attention,
    NestedTensorBlock as Block,
)


def apply_to_layers(function, module, name="", depth_first=True, include_root=False):
    """Recursively traverses a Keras model or layer and applies a function.

    This utility iterates through all `keras.layers.Layer` instances contained
    within a given module, including those nested in lists. For each layer found,
    it calls the provided `function` with the layer instance and its
    hierarchical name (e.g., 'block.0.attention').

    Args:
        function (callable): The function to apply to each layer. It should
            accept two arguments: the layer object and its string name.
        module (keras.layers.Layer): The root Keras Model or Layer to traverse.
        name (str, optional): The base name for the root module.
            Defaults to "".
        depth_first (bool, optional): Controls the traversal order. If True,
            the function is applied to child layers before the parent layer.
            If False, the parent is processed before its children.
            Defaults to True.
        include_root (bool, optional): If True, the function is also applied
            to the initial `module` itself. Defaults to False.

    Returns:
        keras.layers.Layer: The original module, allowing for method chaining.
    """
    if not depth_first and include_root:
        function(module, name)
    for child_name, child_module in module.__dict__.items():
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child_module, layers.Layer):
            named_apply(function, child_module, full_name, depth_first, True)
        elif isinstance(child_module, list) and all(
            isinstance(m, layers.Layer) for m in child_module
        ):
            for i, sub_module in enumerate(child_module):
                named_apply(function, sub_module, f"{full_name}.{i}", depth_first, True)
    if depth_first and include_root:
        function(module, name)
    return module


def init_weights_vit_timm(module, name=""):
    if isinstance(module, layers.Dense):
        standard_deviation = 0.02
        module.kernel.assign(
            initializers.TruncatedNormal(stddev=standard_deviation)(module.kernel.shape)
        )
        if module.bias is not None:
            module.bias.assign(initializers.Zeros()(module.bias.shape))


class BlockChunk(layers.Layer):
    def __init__(self, blocks, **kwargs):
        super().__init__(**kwargs)
        self.blocks = blocks

    def call(self, inputs, training=None):
        for block in self.blocks:
            inputs = block(inputs, training=training)
        return inputs


class DinoVisionTransformer(Model):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        input_channels=3,
        embedding_dimension=768,
        depth=12,
        number_of_heads=12,
        mlp_ratio=4.0,
        use_query_key_value_bias=True,
        use_feedforward_network_bias=True,
        use_projection_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embedding_layer=PatchEmbed,
        activation_layer=layers.Activation("gelu"),
        block_function=Block,
        feedforward_network_layer="mlp",
        block_chunks=1,
        number_of_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        normalization_layer = partial(layers.LayerNormalization, epsilon=1e-6)
        self.number_of_features = self.embedding_dimension = embedding_dimension
        self.number_of_tokens = 1
        self.blockes_number = depth
        self.number_of_heads = number_of_heads
        self.patch_size = patch_size
        self.number_of_register_tokens = number_of_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.feedforward_network_layer = feedforward_network_layer
        self.patch_embedding = embedding_layer(
            img_size=img_size,
            patch_size=patch_size,
            input_channels=input_channels,
            embedding_dimension=embedding_dimension,
        )
        self.init_values = init_values
        number_of_patches = self.patch_embedding.number_of_patches
        self.classification_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embedding_dimension),
            initializer=initializers.RandomNormal(stddev=1e-6),
        )
        self.positional_embedding = self.add_weight(
            name="pos_embed",
            shape=(1, number_of_patches + self.number_of_tokens, embedding_dimension),
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        self.register_tokens = (
            self.add_weight(
                name="register_tokens",
                shape=(1, number_of_register_tokens, embedding_dimension),
                initializer=initializers.RandomNormal(stddev=1e-6),
            )
            if number_of_register_tokens > 0
            else None
        )

        drop_path_rates = (
            [drop_path_rate] * depth
            if drop_path_uniform
            else np.linspace(0.0, drop_path_rate, depth).tolist()
        )

        blocks_list = [
            block_function(
                dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                mlp_ratio=mlp_ratio,
                use_query_key_value_bias=use_query_key_value_bias,
                use_projection_bias=use_projection_bias,
                use_feedforward_network_bias=use_feedforward_network_bias,
                drop_path=drop_path_rates[i],
                normalization_layer=normalization_layer,
                activation_layer=activation_layer,
                feedforward_network_layer=self.feedforward_network_layer,
                init_values=init_values,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]

        if block_chunks > 0:
            self.chunked_blocks = True
            chunksize = depth // block_chunks
            self.blocks = [
                BlockChunk(blocks_list[i : i + chunksize], name=f"chunk_{i//chunksize}")
                for i in range(0, depth, chunksize)
            ]
        else:
            self.chunked_blocks = False
            self.blocks = blocks_list

        self.normalization = normalization_layer(name="norm")
        self.head = layers.Identity(name="head")
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(1, embedding_dimension),
            initializer=initializers.Zeros(),
        )

    def interpolate_positional_encoding(self, x, w, h):
        previous_dtype = x.dtype
        number_of_patches = ops.shape(x)[1] - 1
        N = ops.shape(self.positional_embedding)[1] - 1
        if number_of_patches == N and w == h:
            return self.positional_embedding

        positional_embedding = ops.cast(self.positional_embedding, "float32")
        class_positional_embedding = positional_embedding[:, : self.number_of_tokens]
        patch_positional_embedding = positional_embedding[:, self.number_of_tokens :]
        dimension = ops.shape(x)[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M, "Positional embedding grid must be a perfect square."

        patch_positional_embedding = ops.reshape(
            patch_positional_embedding, (1, M, M, dimension)
        )
        target_w = round((w0 + self.interpolate_offset) / M * M)
        target_h = round((h0 + self.interpolate_offset) / M * M)
        patch_positional_embedding = ops.image.resize(
            patch_positional_embedding,
            size=(target_h, target_w),
            interpolation="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_positional_embedding = ops.slice(
            patch_positional_embedding, [0, 0, 0, 0], [1, h0, w0, dimension]
        )
        patch_positional_embedding = ops.reshape(
            patch_positional_embedding, (1, -1, dimension)
        )
        return ops.cast(
            ops.concatenate(
                [class_positional_embedding, patch_positional_embedding], axis=1
            ),
            previous_dtype,
        )

    def prepare_tokens_with_masks(self, x, masks=None):
        B, H, W, C = ops.shape(x)
        x = self.patch_embedding(x)
        if masks is not None:
            mask_token_expanded = ops.broadcast_to(
                self.mask_token, (B, ops.shape(x)[1], self.embedding_dimension)
            )
            x = ops.where(ops.expand_dimensions(masks, -1), mask_token_expanded, x)
        classification_tokens = ops.broadcast_to(
            self.classification_token, (B, 1, self.embedding_dimension)
        )
        x = ops.concatenate([classification_tokens, x], axis=1)
        x = ops.add(x, self.interpolate_positional_encoding(x, H, W))
        if self.register_tokens is not None:
            register_tokens = ops.broadcast_to(
                self.register_tokens,
                (B, self.number_of_register_tokens, self.embedding_dimension),
            )
            x = ops.concatenate([x[:, :1], register_tokens, x[:, 1:]], axis=1)
        return x

    def forward_features(self, x, masks=None, training=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for block in self.blocks:
            x = block(x, training=training)
        x_normalization = self.normalization(x)
        return {
            "x_normalization_clstoken": x_normalization[:, 0],
            "x_normalization_regtokens": x_normalization[
                :, 1 : self.number_of_register_tokens + 1
            ],
            "x_normalization_patchtokens": x_normalization[
                :, self.number_of_register_tokens + 1 :
            ],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i = [], 0
        total_block_len = self.blockes_number
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        for chunk in self.blocks:
            for block in chunk.blocks:
                x = block(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1

        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in blocks_to_take:
                output.append(x)
        return output

    def get_intermediate_layers(
        self,
        x,
        n=1,
        reshape=False,
        return_class_token=False,
        normalization=True,
    ):
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if normalization:
            outputs = [self.normalization(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.number_of_register_tokens :] for out in outputs]
        if reshape:
            B, H, W, C = ops.shape(x)
            patch_h, patch_w = H // self.patch_size, W // self.patch_size
            outputs = [ops.reshape(out, (B, patch_h, patch_w, -1)) for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def call(self, x, training=None, masks=None):
        is_list = isinstance(x, list)
        if not is_list:
            x_list = [x]
            masks_list = [masks] if masks is not None else [None]
        else:
            x_list = x
            masks_list = masks if masks is not None else [None] * len(x_list)

        if is_list:
            ret = self.forward_features_list(x_list, masks_list)
        else:
            ret = self.forward_features(x_list[0], masks_list[0], training=training)

        if training:
            return ret
        else:
            final_output = ret[0] if is_list else ret
            return self.head(final_output["x_normalization_clstoken"])

    def forward_features_list(self, x_list, masks_list):
        x_list = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for block in self.blocks:
            x_list = [block(x) for x in x_list]
        all_x = x_list
        output = []
        for x_i, masks_i in zip(all_x, masks_list):
            x_normalization = self.normalization(x_i)
            output.append(
                {
                    "x_normalization_clstoken": x_normalization[:, 0],
                    "x_normalization_regtokens": x_normalization[
                        :, 1 : self.number_of_register_tokens + 1
                    ],
                    "x_normalization_patchtokens": x_normalization[
                        :, self.number_of_register_tokens + 1 :
                    ],
                    "x_prenorm": x_i,
                    "masks": masks_i,
                }
            )
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_size": self.patch_embedding.img_size,
                "patch_size": self.patch_size,
                "input_channels": self.patch_embedding.input_channels,
                "embedding_dimension": self.embedding_dimension,
                "depth": self.blockes_number,
                "number_of_heads": self.number_of_heads,
                "mlp_ratio": 4.0,
                "use_query_key_value_bias": True,
                "use_feedforward_network_bias": True,
                "use_projection_bias": True,
                "drop_path_rate": 0.0,
                "init_values": self.init_values,
                "number_of_register_tokens": self.number_of_register_tokens,
                "interpolate_antialias": self.interpolate_antialias,
                "interpolate_offset": self.interpolate_offset,
                "feedforward_network_layer": self.feedforward_network_layer,
            }
        )
        return config


def vit_small(patch_size=16, number_of_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embedding_dimension=384,
        depth=12,
        number_of_heads=6,
        mlp_ratio=4,
        block_function=partial(Block, attention_class=Attention),
        number_of_register_tokens=number_of_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, number_of_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embedding_dimension=768,
        depth=12,
        number_of_heads=12,
        mlp_ratio=4,
        block_function=partial(Block, attention_class=Attention),
        number_of_register_tokens=number_of_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, number_of_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embedding_dimension=1024,
        depth=24,
        number_of_heads=16,
        mlp_ratio=4,
        block_function=partial(Block, attention_class=Attention),
        number_of_register_tokens=number_of_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(
    patch_size=16,
    number_of_register_tokens=0,
    feedforward_network_layer="swiglu",
    **kwargs,
):
    """
    Close to ViT-giant, with embed-dimension 1536 and 24 heads => embed-dimension per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embedding_dimension=1536,
        depth=40,
        number_of_heads=24,
        mlp_ratio=4,
        block_function=partial(Block, attention_class=Attention),
        number_of_register_tokens=number_of_register_tokens,
        feedforward_network_layer=feedforward_network_layer,
        **kwargs,
    )
    return model
