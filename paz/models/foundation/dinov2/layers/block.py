import keras
from keras import ops
from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import MLP


def drop_add_residual_stochastic_depth(
    x,
    residual_func,
    sample_drop_ratio=0.0,
):
    b = ops.shape(x)[0]

    subset_float = ops.multiply(ops.cast(b, "float32"), (1.0 - sample_drop_ratio))
    sample_subset_size = ops.maximum(ops.cast(subset_float, "int32"), 1)

    all_indices = ops.arange(b, dtype="int32")
    brange = keras.random.shuffle(all_indices)[:sample_subset_size]

    x_subset = ops.take(x, brange, axis=0)

    residual = residual_func(x_subset)

    original_shape = ops.shape(x)
    x_flat = ops.reshape(x, (b, -1))
    residual_flat = ops.reshape(residual, (sample_subset_size, -1))

    b_float = ops.cast(b, x.dtype)
    sample_subset_size_float = ops.cast(sample_subset_size, x.dtype)
    residual_scale_factor = ops.divide(b_float, sample_subset_size_float)

    scaled_residual = ops.multiply(residual_flat, residual_scale_factor)

    scatter_indices = ops.expand_dims(brange, axis=1)

    existing_values = ops.take(x_flat, brange, axis=0)
    updated_values = ops.add(existing_values, scaled_residual)

    x_plus_residual_flat = ops.scatter_update(x_flat, scatter_indices, updated_values)

    return ops.reshape(x_plus_residual_flat, original_shape)


def get_branges_scales(x, sample_drop_ratio=0.0):

    b = ops.shape(x)[0]

    b_float = ops.cast(b, "float32")
    subset_size_float = ops.multiply(b_float, (1.0 - sample_drop_ratio))

    sample_subset_size = ops.maximum(ops.cast(subset_size_float, "int32"), 1)

    all_indices = ops.arange(b, dtype="int32")
    brange = keras.random.shuffle(all_indices)[:sample_subset_size]

    sample_subset_size_float = ops.cast(sample_subset_size, dtype="float32")
    residual_scale_factor = ops.divide(b_float, sample_subset_size_float)

    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    residual = ops.cast(residual, dtype=x.dtype)

    if scaling_vector is None:
        original_shape = ops.shape(x)
        b = original_shape[0]

        x_flat = ops.reshape(x, (b, -1))
        residual_flat = ops.reshape(residual, (ops.shape(residual)[0], -1))

        scaled_residual = ops.multiply(residual_flat, residual_scale_factor)

        existing_values = ops.take(x_flat, brange, axis=0)
        updated_values = ops.add(existing_values, scaled_residual)
        scatter_indices = ops.expand_dims(brange, axis=1)
        x_plus_residual_flat = ops.scatter_update(x_flat, scatter_indices, updated_values)

        return ops.reshape(x_plus_residual_flat, original_shape)

    else:
        element_wise_scaled_residual = ops.multiply(residual, scaling_vector)

        final_residual = ops.multiply(element_wise_scaled_residual, residual_scale_factor)

        existing_values = ops.take(x, brange, axis=0)
        updated_values = ops.add(existing_values, final_residual)
        scatter_indices = ops.expand_dims(brange, axis=1)
        x_plus_residual = ops.scatter_update(x, scatter_indices, updated_values)

        return x_plus_residual


attn_bias_cache = {}


def _index_select_cat(tensors, branges):
    selected_tensors = [ops.take(tensor, brange, axis=0) for tensor, brange in zip(tensors, branges)]
    return ops.concatenate(selected_tensors, axis=0)


def _create_manual_block_diagonal(seqlens, dtype):
    num_blocks = len(seqlens)
    rows_of_blocks = []
    for i in range(num_blocks):
        block_row = []
        for j in range(num_blocks):
            rows = seqlens[i]
            cols = seqlens[j]
            if i == j:
                block = ops.ones((rows, cols), dtype=dtype)
            else:
                block = ops.zeros((rows, cols), dtype=dtype)
            block_row.append(block)
        rows_of_blocks.append(ops.concatenate(block_row, axis=1))
    return ops.concatenate(rows_of_blocks, axis=0)


def get_attn_bias_and_cat(x_list, branges=None):
    if branges is not None:
        batch_sizes = [ops.shape(b)[0] for b in branges]
    else:
        batch_sizes = [ops.shape(x)[0] for x in x_list]

    all_shapes = tuple((b, ops.shape(x)[1]) for b, x in zip(batch_sizes, x_list))

    if all_shapes not in attn_bias_cache:
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            seq_len = ops.shape(x)[1]
            for _ in range(b):
                seqlens.append(seq_len)

        attn_bias = _create_manual_block_diagonal(seqlens, dtype=x_list[0].dtype)
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        tensors_flat = [ops.reshape(x, (ops.shape(x)[0], -1)) for x in x_list]
        concatenated_flat = _index_select_cat(tensors_flat, branges)
        cat_tensors = ops.reshape(concatenated_flat, (1, -1, ops.shape(x_list[0])[-1]))
    else:
        tensors_bs1 = [ops.reshape(x, (1, -1, ops.shape(x)[-1])) for x in x_list]
        cat_tensors = ops.concatenate(tensors_bs1, axis=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list,
    residual_func,
    sample_drop_ratio=0.0,
    scaling_vector=None,
):
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    attn_mask, x_cat = get_attn_bias_and_cat(x_list, branges)

    residual_cat = residual_func(x_cat, attn_bias=attn_mask)

    split_lengths = []
    sub_batch_shapes = []
    for x, brange in zip(x_list, branges):
        sub_batch_size = ops.shape(brange)[0]
        seq_len = ops.shape(x)[1]
        dim = ops.shape(x)[2]
        split_lengths.append(sub_batch_size * seq_len)
        sub_batch_shapes.append((sub_batch_size, seq_len, dim))

    residual_cat_flat = ops.reshape(residual_cat, (-1, ops.shape(residual_cat)[-1]))

    split_residuals_flat = ops.split(residual_cat_flat, split_lengths, axis=0)

    residual_list = [ops.reshape(res, shape) for res, shape in zip(split_residuals_flat, sub_batch_shapes)]

    outputs = []
    for x, brange, residual, res_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        output = add_residual(x, brange, residual, res_scale_factor, scaling_vector)
        outputs.append(output)

    return outputs


class Block(keras.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=keras.activations.gelu,
        norm_layer=keras.layers.LayerNormalization,
        attn_class=Attention,
        ffn_layer=MLP,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = norm_layer()
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            projection_bias=proj_bias,
            attention_drop_rate=attn_drop,
            projection_drop_rate=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else keras.layers.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else keras.layers.Identity()

        self.norm2 = norm_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else keras.layers.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else keras.layers.Identity()

        self.sample_drop_ratio = drop_path

    def call(self, x, training=None):
        # Attention Block
        shortcut1 = x
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, training=training)
        x_attn = self.ls1(x_attn)
        x_attn = self.drop_path1(x_attn, training=training)
        x = shortcut1 + x_attn

        # MLP Block
        shortcut2 = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm, training=training)
        x_mlp = self.ls2(x_mlp)
        x_mlp = self.drop_path2(x_mlp, training=training)
        x = shortcut2 + x_mlp

        return x


class CausalAttentionBlock(keras.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        ls_init_value=None,
        is_causal=True,
        act_layer=keras.activations.gelu,
        norm_layer=keras.layers.LayerNormalization,
        dropout_prob=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.is_causal = is_causal

        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else keras.layers.Identity()
        self.attention_norm = norm_layer()
        self.attention = Attention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer()
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = MLP(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )
        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else keras.layers.Identity()

    def init_weights(
        self,
        init_attn_std=None,
        init_proj_std=None,
        init_fc_std=None,
        factor=1.0,
    ) -> None:

        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5

        if hasattr(self.attention, "init_weights"):
            self.attention.init_weights(init_attn_std, init_proj_std)

        self.attention_norm.gamma.assign(ops.ones(self.attention_norm.gamma.shape))
        self.attention_norm.beta.assign(ops.zeros(self.attention_norm.beta.shape))
        self.ffn_norm.gamma.assign(ops.ones(self.ffn_norm.gamma.shape))
        self.ffn_norm.beta.assign(ops.zeros(self.ffn_norm.beta.shape))

        self.feed_forward.fc1.kernel.assign(
            keras.random.normal(self.feed_forward.fc1.kernel.shape, stddev=init_fc_std)
        )
        self.feed_forward.fc2.kernel.assign(
            keras.random.normal(self.feed_forward.fc2.kernel.shape, stddev=init_proj_std)
        )

    def call(self, x):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), is_causal=self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn


class NestedTensorBlock(Block):
    def call_nested(self, x_list, training=None):
        if training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x, attn_bias=None):
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x, attn_bias=None):
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
            )
            return x_list

        else:

            def attn_residual_func(x, attn_bias=None):
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x, attn_bias=None):
                return self.ls2(self.mlp(self.norm2(x)))

            attn_mask, x_cat = get_attn_bias_and_cat(x_list)

            x_cat = x_cat + attn_residual_func(x_cat, attn_bias=attn_mask)
            x_cat = x_cat + ffn_residual_func(x_cat)

            split_lengths = [ops.shape(t)[0] * ops.shape(t)[1] for t in x_list]
            x_cat_flat = ops.reshape(x_cat, (-1, ops.shape(x_cat)[-1]))
            split_x_flat = ops.split(x_cat_flat, split_lengths, axis=0)
            original_shapes = [ops.shape(t) for t in x_list]
            output_list = [ops.reshape(tensor, shape) for tensor, shape in zip(split_x_flat, original_shapes)]

            return output_list

    def call(self, x_or_x_list, training=None):
        if isinstance(x_or_x_list, list):
            return self.call_nested(x_or_x_list, training=training)
        elif hasattr(x_or_x_list, "shape") and hasattr(x_or_x_list, "dtype"):
            return super().call(x_or_x_list, training=training)
        else:
            raise TypeError(f"Input must be a tensor or a list of tensors, but got {type(x_or_x_list)}")
