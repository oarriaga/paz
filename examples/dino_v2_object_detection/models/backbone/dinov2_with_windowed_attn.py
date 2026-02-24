import math
from functools import partial

import keras
from keras import layers, ops, initializers, Model

from paz.models.foundation.dinov2.layers import (
    MLP,
    LayerScale,
    DropPath,
    SwiGLUFFNFused,
    Attention,
)


@keras.saving.register_keras_serializable(package="backbone")
class WindowedDinov2PatchEmbeddings(layers.Layer):
    """
    Construct the CLS token, mask token, register tokens, position and
    patch embeddings.  Optionally reshapes the output into windows if
    num_windows > 1.

    Corresponds to PyTorch:
      - Dinov2WithRegistersPatchEmbeddings
      - WindowedDinov2WithRegistersEmbeddings
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_register_tokens=0,
        num_windows=1,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = (
            image_size
            if isinstance(image_size, (tuple, list))
            else (image_size, image_size)
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens
        self.num_windows = num_windows
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        num_patches = (self.image_size[1] // self.patch_size[1]) * (
            self.image_size[0] // self.patch_size[0]
        )
        self.num_patches = num_patches

        # Patch Embedding (Conv2D) — channels-last in Keras
        self.projection = layers.Conv2D(
            hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="patch_embeddings_projection",
            padding="valid",
        )

        # CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, hidden_size),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        # Position embeddings (num_patches + 1 for CLS)
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, num_patches + 1, hidden_size),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        # Register tokens
        if num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                name="register_tokens",
                shape=(1, num_register_tokens, hidden_size),
                initializer=initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
            )
        else:
            self.register_tokens = None

        self.dropout = layers.Dropout(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": list(self.image_size),
                "patch_size": list(self.patch_size),
                "num_channels": self.num_channels,
                "hidden_size": self.hidden_size,
                "num_register_tokens": self.num_register_tokens,
                "num_windows": self.num_windows,
                "interpolate_antialias": self.interpolate_antialias,
                "interpolate_offset": self.interpolate_offset,
            }
        )
        return config

    def interpolate_pos_encoding(self, embeddings, height, width):
        """Interpolate position encodings for arbitrary resolution."""
        num_patches = ops.shape(embeddings)[1] - 1
        num_positions = ops.shape(self.position_embeddings)[1] - 1

        if (
            num_patches == num_positions
            and height == self.image_size[0]
            and width == self.image_size[1]
        ):
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = ops.shape(embeddings)[-1]

        h0 = height // self.patch_size[0]
        w0 = width // self.patch_size[1]

        sqrt_num_positions = int(math.sqrt(num_positions))

        # Reshape to (1, H, W, D) for Keras image resize
        patch_pos_embed = ops.reshape(
            patch_pos_embed, (1, sqrt_num_positions, sqrt_num_positions, dim)
        )

        patch_pos_embed = ops.image.resize(
            patch_pos_embed,
            size=(h0, w0),
            interpolation="bicubic",
            antialias=self.interpolate_antialias,
        )

        patch_pos_embed = ops.reshape(patch_pos_embed, (1, -1, dim))
        return ops.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def call(self, pixel_values, training=None):
        batch_size = ops.shape(pixel_values)[0]
        height = ops.shape(pixel_values)[1]
        width = ops.shape(pixel_values)[2]

        # 1. Patch Embeddings
        embeddings = self.projection(pixel_values)  # (B, H', W', C)
        embeddings = ops.reshape(
            embeddings, (batch_size, -1, self.hidden_size)
        )  # (B, N, C)

        # 2. Add CLS Token
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.hidden_size))
        embeddings = ops.concatenate([cls_tokens, embeddings], axis=1)

        # 3. Add Positional Encoding
        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, height, width
        )

        # 4. Windowing Logic
        if self.num_windows > 1:
            num_h_patches = height // self.patch_size[0]
            num_w_patches = width // self.patch_size[1]

            cls_token_with_pos = embeddings[:, :1]
            pixel_tokens_with_pos = embeddings[:, 1:]

            pixel_tokens_with_pos = ops.reshape(
                pixel_tokens_with_pos,
                (batch_size, num_h_patches, num_w_patches, -1),
            )

            # Padding for divisibility by num_windows
            pad_h = (
                self.num_windows - num_h_patches % self.num_windows
            ) % self.num_windows
            pad_w = (
                self.num_windows - num_w_patches % self.num_windows
            ) % self.num_windows

            if pad_h > 0 or pad_w > 0:
                pixel_tokens_with_pos = ops.pad(
                    pixel_tokens_with_pos,
                    [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
                )
                num_h_patches = num_h_patches + pad_h
                num_w_patches = num_w_patches + pad_w

            num_w_patches_per_window = num_w_patches // self.num_windows
            num_h_patches_per_window = num_h_patches // self.num_windows

            windowed_pixel_tokens = ops.reshape(
                pixel_tokens_with_pos,
                (
                    batch_size,
                    self.num_windows,
                    num_h_patches_per_window,
                    self.num_windows,
                    num_w_patches_per_window,
                    -1,
                ),
            )
            windowed_pixel_tokens = ops.transpose(
                windowed_pixel_tokens, (0, 1, 3, 2, 4, 5)
            )

            windowed_pixel_tokens = ops.reshape(
                windowed_pixel_tokens,
                (
                    batch_size * (self.num_windows**2),
                    num_h_patches_per_window * num_w_patches_per_window,
                    -1,
                ),
            )

            windowed_cls_token = ops.broadcast_to(
                ops.expand_dims(cls_token_with_pos, 1),
                (batch_size, self.num_windows**2, 1, self.hidden_size),
            )
            windowed_cls_token = ops.reshape(
                windowed_cls_token,
                (batch_size * (self.num_windows**2), 1, self.hidden_size),
            )

            embeddings = ops.concatenate(
                [windowed_cls_token, windowed_pixel_tokens], axis=1
            )

        # 5. Register Tokens
        if self.num_register_tokens > 0:
            current_batch_size = ops.shape(embeddings)[0]
            register_tokens = ops.broadcast_to(
                self.register_tokens,
                (current_batch_size, self.num_register_tokens, self.hidden_size),
            )
            embeddings = ops.concatenate(
                [embeddings[:, :1], register_tokens, embeddings[:, 1:]], axis=1
            )

        embeddings = self.dropout(embeddings, training=training)
        return embeddings


@keras.saving.register_keras_serializable(package="backbone")
class WindowedDinov2Layer(layers.Layer):
    """
    DINOv2 Layer with optional Windowed Attention support.
    Corresponds to PyTorch WindowedDinov2WithRegistersLayer.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        use_swiglu_ffn=False,
        num_windows=1,
        init_values=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_windows = num_windows
        self._num_attention_heads = num_attention_heads
        self._mlp_ratio = mlp_ratio
        self._drop_path_rate = drop_path_rate
        self._layer_norm_eps = layer_norm_eps
        self._use_swiglu_ffn = use_swiglu_ffn
        self._init_values = init_values

        self.norm1 = layers.LayerNormalization(epsilon=layer_norm_eps, name="norm1")

        self.attention = Attention(
            dimension=hidden_size,
            number_of_heads=num_attention_heads,
            use_query_key_value_bias=True,
            use_projection_bias=True,
            name="attention",
        )

        self.layer_scale1 = LayerScale(hidden_size, init_values=init_values, name="ls1")
        self.drop_path1 = (
            DropPath(drop_path_rate, name="drop_path1")
            if drop_path_rate > 0
            else layers.Identity()
        )

        self.norm2 = layers.LayerNormalization(epsilon=layer_norm_eps, name="norm2")

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu_ffn:
            self.mlp = SwiGLUFFNFused(
                input_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                use_bias=True,
                name="mlp",
            )
        else:
            self.mlp = MLP(
                input_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                activation_layer=layers.Activation("gelu"),
                use_bias=True,
            )

        self.layer_scale2 = LayerScale(hidden_size, init_values=init_values, name="ls2")
        self.drop_path2 = (
            DropPath(drop_path_rate, name="drop_path2")
            if drop_path_rate > 0
            else layers.Identity()
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self._num_attention_heads,
                "mlp_ratio": self._mlp_ratio,
                "drop_path_rate": self._drop_path_rate,
                "layer_norm_eps": self._layer_norm_eps,
                "use_swiglu_ffn": self._use_swiglu_ffn,
                "num_windows": self.num_windows,
                "init_values": self._init_values,
            }
        )
        return config

    def call(self, hidden_states, training=None, run_full_attention=False):
        shortcut = hidden_states

        x = self.norm1(hidden_states)

        # Windowed → full attention reshape
        if run_full_attention and self.num_windows > 1:
            B_eff = ops.shape(x)[0]
            N_win = ops.shape(x)[1]
            C = ops.shape(x)[2]
            num_windows_squared = self.num_windows**2

            x = ops.reshape(
                x, (B_eff // num_windows_squared, num_windows_squared * N_win, C)
            )

        x_attn = self.attention(x, training=training)

        if run_full_attention and self.num_windows > 1:
            x_attn = ops.reshape(x_attn, (B_eff, N_win, C))

        x_attn = self.layer_scale1(x_attn)
        x = shortcut + self.drop_path1(x_attn, training=training)

        # FFN
        shortcut = x
        x = self.norm2(x)
        x_mlp = self.mlp(x, training=training)
        x_mlp = self.layer_scale2(x_mlp)
        x = shortcut + self.drop_path2(x_mlp, training=training)

        return x


@keras.saving.register_keras_serializable(package="backbone")
class WindowedDinov2Encoder(layers.Layer):
    """
    Stacks WindowedDinov2Layer blocks.
    Corresponds to PyTorch WindowedDinov2WithRegistersEncoder.
    """

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        use_swiglu_ffn=False,
        num_windows=1,
        window_block_indexes=None,
        init_values=1e-5,
        out_feature_indexes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self._hidden_size = hidden_size
        self._num_attention_heads = num_attention_heads
        self._mlp_ratio = mlp_ratio
        self._drop_path_rate = drop_path_rate
        self._layer_norm_eps = layer_norm_eps
        self._use_swiglu_ffn = use_swiglu_ffn
        self._num_windows = num_windows
        self._init_values = init_values
        self.window_block_indexes = (
            window_block_indexes
            if window_block_indexes is not None
            else list(range(num_hidden_layers))
        )
        self.out_feature_indexes = (
            out_feature_indexes if out_feature_indexes is not None else []
        )

        # Stochastic Depth Decay Rule
        dpr = [
            (
                x * drop_path_rate / (num_hidden_layers - 1)
                if num_hidden_layers > 1
                else 0.0
            )
            for x in range(num_hidden_layers)
        ]

        self.encoder_layers = []
        for i in range(num_hidden_layers):
            self.encoder_layers.append(
                WindowedDinov2Layer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path_rate=dpr[i],
                    layer_norm_eps=layer_norm_eps,
                    use_swiglu_ffn=use_swiglu_ffn,
                    num_windows=num_windows,
                    init_values=init_values,
                    name=f"layer_{i}",
                )
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self._hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self._num_attention_heads,
                "mlp_ratio": self._mlp_ratio,
                "drop_path_rate": self._drop_path_rate,
                "layer_norm_eps": self._layer_norm_eps,
                "use_swiglu_ffn": self._use_swiglu_ffn,
                "num_windows": self._num_windows,
                "window_block_indexes": self.window_block_indexes,
                "init_values": self._init_values,
                "out_feature_indexes": self.out_feature_indexes,
            }
        )
        return config

    def call(self, hidden_states, training=None):
        all_hidden_states = []

        for i, layer_module in enumerate(self.encoder_layers):
            run_full_attention = i not in self.window_block_indexes

            hidden_states = layer_module(
                hidden_states,
                training=training,
                run_full_attention=run_full_attention,
            )
            all_hidden_states.append(hidden_states)

        return all_hidden_states


@keras.saving.register_keras_serializable(package="backbone")
class WindowedDinov2Model(Model):
    """
    Full DINOv2 Model: Embeddings + Encoder + LayerNorm.
    Corresponds to PyTorch WindowedDinov2WithRegistersModel.
    """

    def __init__(
        self,
        image_size=518,
        patch_size=14,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        use_swiglu_ffn=False,
        num_register_tokens=0,
        num_windows=1,
        window_block_indexes=None,
        init_values=1e-5,
        out_feature_indexes=None,
        interpolate_antialias=False,
        name="dinov2_model",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.hidden_size = hidden_size
        self._image_size = image_size
        self._num_channels = num_channels
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._mlp_ratio = mlp_ratio
        self._drop_path_rate = drop_path_rate
        self._layer_norm_eps = layer_norm_eps
        self._use_swiglu_ffn = use_swiglu_ffn
        self._window_block_indexes = window_block_indexes
        self._init_values = init_values
        self._out_feature_indexes = out_feature_indexes
        self._interpolate_antialias = interpolate_antialias

        self.embeddings = WindowedDinov2PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
            num_windows=num_windows,
            interpolate_antialias=interpolate_antialias,
            name="embeddings",
        )

        self.encoder = WindowedDinov2Encoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
            use_swiglu_ffn=use_swiglu_ffn,
            num_windows=num_windows,
            window_block_indexes=window_block_indexes,
            init_values=init_values,
            out_feature_indexes=out_feature_indexes,
            name="encoder",
        )

        self.layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layernorm"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self._image_size,
                "patch_size": self.patch_size,
                "num_channels": self._num_channels,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self._num_hidden_layers,
                "num_attention_heads": self._num_attention_heads,
                "mlp_ratio": self._mlp_ratio,
                "drop_path_rate": self._drop_path_rate,
                "layer_norm_eps": self._layer_norm_eps,
                "use_swiglu_ffn": self._use_swiglu_ffn,
                "num_register_tokens": self.num_register_tokens,
                "num_windows": self.num_windows,
                "window_block_indexes": self._window_block_indexes,
                "init_values": self._init_values,
                "out_feature_indexes": self._out_feature_indexes,
                "interpolate_antialias": self._interpolate_antialias,
            }
        )
        return config

    @property
    def blocks(self):
        return self.encoder.encoder_layers

    @property
    def normalization(self):
        return self.layernorm

    @property
    def window_block_indexes(self):
        return self.encoder.window_block_indexes

    @property
    def number_of_register_tokens(self):
        return self.num_register_tokens

    def call(self, pixel_values, training=None):
        embedding_output = self.embeddings(pixel_values, training=training)
        encoder_outputs = self.encoder(embedding_output, training=training)

        sequence_output = encoder_outputs[-1]
        sequence_output = self.layernorm(sequence_output)

        return sequence_output, encoder_outputs

    def prepare_tokens_with_masks(self, inputs):
        return self.embeddings(inputs)

    @property
    def embedding_dimension(self):
        return self.hidden_size


# ─────────────────────────────────────────────────────────────
#  Builder Functions
# ─────────────────────────────────────────────────────────────


def dinov2_windowed_small(
    img_size=518,
    patch_size=14,
    number_of_register_tokens=0,
    num_windows=37,
    window_block_indexes=None,
    **kwargs,
):
    if window_block_indexes is None:
        window_block_indexes = (
            list(range(0, 2))
            + list(range(3, 5))
            + list(range(6, 8))
            + list(range(9, 11))
        )
    return WindowedDinov2Model(
        image_size=img_size,
        patch_size=patch_size,
        num_register_tokens=number_of_register_tokens,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        mlp_ratio=4.0,
        num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        **kwargs,
    )


def dinov2_windowed_base(
    img_size=518,
    patch_size=14,
    number_of_register_tokens=0,
    num_windows=37,
    window_block_indexes=None,
    **kwargs,
):
    if window_block_indexes is None:
        window_block_indexes = (
            list(range(0, 2))
            + list(range(3, 5))
            + list(range(6, 8))
            + list(range(9, 11))
        )
    return WindowedDinov2Model(
        image_size=img_size,
        patch_size=patch_size,
        num_register_tokens=number_of_register_tokens,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        **kwargs,
    )


def dinov2_windowed_large(
    img_size=518,
    patch_size=14,
    number_of_register_tokens=0,
    num_windows=37,
    window_block_indexes=None,
    **kwargs,
):
    if window_block_indexes is None:
        window_block_indexes = (
            list(range(0, 5))
            + list(range(6, 11))
            + list(range(12, 17))
            + list(range(18, 23))
        )
    return WindowedDinov2Model(
        image_size=img_size,
        patch_size=patch_size,
        num_register_tokens=number_of_register_tokens,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4.0,
        num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        **kwargs,
    )


def dinov2_windowed_giant(
    img_size=518,
    patch_size=14,
    number_of_register_tokens=0,
    num_windows=37,
    window_block_indexes=None,
    **kwargs,
):
    if window_block_indexes is None:
        window_block_indexes = list(range(0, 39))
    return WindowedDinov2Model(
        image_size=img_size,
        patch_size=patch_size,
        num_register_tokens=number_of_register_tokens,
        hidden_size=1536,
        num_hidden_layers=40,
        num_attention_heads=24,
        mlp_ratio=4.0,
        use_swiglu_ffn=True,
        num_windows=num_windows,
        window_block_indexes=window_block_indexes,
        **kwargs,
    )
