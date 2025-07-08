import keras
from keras import ops
from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import MLP


def get_random_subset_of_indices(batch_size, drop_ratio):
    """Selects a random subset of indices from the batch."""
    num_samples_to_keep_float = ops.multiply(ops.cast(batch_size, "float32"), (1.0 - drop_ratio))

    num_samples_to_keep = ops.maximum(ops.cast(num_samples_to_keep_float, "int32"), 1)

    all_possible_indices = ops.arange(batch_size, dtype="int32")

    selected_indices = keras.random.shuffle(all_possible_indices)[:num_samples_to_keep]

    return selected_indices


def scale_and_add_residual(
    original_input_tensor,
    selected_sample_indices,
    residual_to_add,
    residual_scale_factor,
):
    """Scales a residual and adds it to a subset of the input tensor."""
    original_tensor_shape = ops.shape(original_input_tensor)
    batch_size = original_tensor_shape[0]

    flattened_input_tensor = ops.reshape(original_input_tensor, (batch_size, -1))
    flattened_residual = ops.reshape(residual_to_add, (ops.shape(selected_sample_indices)[0], -1))

    scaled_flattened_residual = ops.multiply(flattened_residual, residual_scale_factor)

    scatter_indices = ops.expand_dims(selected_sample_indices, axis=1)

    existing_values_at_indices = ops.take(flattened_input_tensor, selected_sample_indices, axis=0)

    updated_values_for_subset = ops.add(existing_values_at_indices, scaled_flattened_residual)

    updated_flattened_tensor = ops.scatter_update(
        flattened_input_tensor, scatter_indices, updated_values_for_subset
    )

    return ops.reshape(updated_flattened_tensor, original_tensor_shape)


def drop_add_residual_stochastic_depth(
    input_tensor,
    apply_residual,
    sample_drop_ratio=0.0,
):
    """Applies a residual function to a random subset of a batch (Stochastic Depth).

    This function implements a form of Stochastic Depth, a regularization technique.
    Instead of applying the residual function to the entire input tensor, it is
    applied only to a randomly selected subset of the samples in the batch. The
    resulting residual is then scaled to preserve the expected value of the output
    before being added back to the corresponding original samples.
    Samples not selected for the residual computation are passed through
    unchanged (identity path).

    Args:
        input_tensor: The input tensor, where the first dimension is the batch size.
        apply_residual: A callable (e.g., a layer or model) that computes the
            residual transformation. It will be called on a subset of `input_tensor`.
        sample_drop_ratio: The probability of dropping a sample from the
            residual path. A value of 0.0 means no samples are dropped
            (standard residual connection), while a value of 1.0 would mean all
            samples are dropped.

    Returns:
        A tensor with the same shape as the input `input_tensor`, where the scaled
        residual has been added to a random subset of the batch samples.
    """
    batch_size = ops.shape(input_tensor)[0]

    selected_sample_indices = get_random_subset_of_indices(batch_size, sample_drop_ratio)

    input_subset = ops.take(input_tensor, selected_sample_indices, axis=0)
    residual_from_subset = apply_residual(input_subset)

    batch_size_as_float = ops.cast(batch_size, input_tensor.dtype)
    num_selected_samples_as_float = ops.cast(ops.shape(selected_sample_indices)[0], input_tensor.dtype)
    residual_scale_factor = ops.divide(batch_size_as_float, num_selected_samples_as_float)
    final_output_tensor = scale_and_add_residual(
        input_tensor,
        selected_sample_indices,
        residual_from_subset,
        residual_scale_factor,
    )
    return final_output_tensor


def get_stochastic_depth_inputs(x, sample_drop_ratio=0.0):
    """Generates random indices and a scaling factor for a batch subset.

    This helper function supports stochastic depth by selecting a random subset
    of samples from a batch to be processed. It calculates the indices for this
    subset and a corresponding scaling factor needed to preserve the expected
    value of the operation.

    Args:
        x: The input tensor. The batch size is inferred from its first dimension.
        sample_drop_ratio: The probability of dropping a sample from processing.
            A value of 0.0 means no samples are dropped.

    Returns:
        A tuple containing two tensors:
            - selected_sample_indices: An int32 tensor of randomly selected indices.
            - residual_scale_factor: A float32 scalar tensor used for scaling.
    """
    batch_size = ops.shape(x)[0]

    batch_size_as_float = ops.cast(batch_size, "float32")
    subset_size_float = ops.multiply(batch_size_as_float, (1.0 - sample_drop_ratio))

    sample_subset_size = ops.maximum(ops.cast(subset_size_float, "int32"), 1)

    all_indices = ops.arange(batch_size, dtype="int32")
    selected_sample_indices = keras.random.shuffle(all_indices)[:sample_subset_size]

    sample_subset_size_float = ops.cast(sample_subset_size, dtype="float32")
    residual_scale_factor = ops.divide(batch_size_as_float, sample_subset_size_float)

    return selected_sample_indices, residual_scale_factor


def add_residual(x, selected_sample_indices, residual, residual_scale_factor, scaling_vector=None):
    """Adds a scaled residual to a specified subset of samples in a tensor.

    This function updates a subset of the input tensor `x` by adding a `residual`
    value. The subset is determined by the `selected_sample_indices` indices. The function has two
    modes for scaling the residual before addition, controlled by the presence of
    the `scaling_vector`.

    - If `scaling_vector` is None, the entire residual is scaled by a single
      scalar `residual_scale_factor`.
    - If `scaling_vector` is provided, the residual undergoes an element-wise
      multiplication with `scaling_vector` followed by scaling with
      `residual_scale_factor`.

    Args:
        x: The input tensor to which the residual will be added.
        selected_sample_indices: A tensor of indices specifying which samples in the batch
            dimension of `x` to update.
        residual: The residual tensor to add to the subset of `x`. Its batch
            dimension should match the size of `selected_sample_indices`.
        residual_scale_factor: A scalar float used to scale the residual.
        scaling_vector: An optional tensor for element-wise scaling of the
            `residual`. If provided, its shape must be broadcastable to the
            shape of `residual`.

    Returns:
        A new tensor with the same shape as `x`, containing the result of the
        residual addition on the specified subset.
    """
    residual = ops.cast(residual, dtype=x.dtype)

    if scaling_vector is not None:
        element_wise_scaled_residual = ops.multiply(residual, scaling_vector)
        final_residual = ops.multiply(element_wise_scaled_residual, residual_scale_factor)
    else:
        final_residual = ops.multiply(residual, residual_scale_factor)

    existing_values = ops.take(x, selected_sample_indices, axis=0)
    updated_values = ops.add(existing_values, final_residual)
    scatter_indices = ops.expand_dims(selected_sample_indices, axis=1)
    x_plus_residual = ops.scatter_update(x, scatter_indices, updated_values)

    return x_plus_residual


attention_bias_cache = {}


def select_and_concatenate_tensors_by_indices(list_of_tensors, list_of_indices_per_tensor):
    """Selects samples from a list of tensors based on a corresponding list of indices
    and concatenates the results into a single tensor.

    This function iterates through a list of tensors and a parallel list of index
    groups. For each tensor, it uses the corresponding indices to select a subset
    of its samples. All these selected subsets are then combined into a single
    output tensor.

    Args:
        list_of_tensors: A list of tensors from which to select samples.
        list_of_indices_per_tensor: A list where each element is a tensor of indices
            corresponding to the tensor at the same position in `list_of_tensors`.

    Returns:
        A single tensor containing the concatenated selected samples.
    """
    selected_tensor_slices = []

    for current_tensor, current_indices in zip(list_of_tensors, list_of_indices_per_tensor):
        selected_slice = ops.take(current_tensor, current_indices, axis=0)

        selected_tensor_slices.append(selected_slice)

    final_concatenated_tensor = ops.concatenate(selected_tensor_slices, axis=0)

    return final_concatenated_tensor


def create_block_diagonal_matrix(block_sizes, data_type):
    """Constructs a block diagonal matrix from a list of block sizes.

    This function creates a larger matrix composed of smaller square matrices (blocks)
    placed along the main diagonal. Each block on the diagonal is filled with ones,
    and all other off-diagonal blocks are filled with zeros.

    For example, if block_sizes is [2, 3], the function will produce a 5x5
    matrix with a 2x2 block of ones and a 3x3 block of ones on the diagonal.

    Args:
        block_sizes (list[int]): A list of integers, where each integer specifies
            the size (height and width) of a square block of ones to be placed
            on the diagonal.
        data_type: The data type for the elements of the resulting matrix
            (e.g., a NumPy or TensorFlow dtype).

    Returns:
        A 2D matrix (compatible with the 'ops' library used) representing the
        final block diagonal matrix.
    """
    number_of_blocks = len(block_sizes)
    concatenated_block_rows = []
    for row_index in range(number_of_blocks):
        current_block_row = []
        for column_index in range(number_of_blocks):
            number_of_rows_in_block = block_sizes[row_index]
            number_of_columns_in_block = block_sizes[column_index]
            matrix_block_shape = (number_of_rows_in_block, number_of_columns_in_block)
            if row_index == column_index:
                matrix_block = ops.ones(matrix_block_shape, dtype=data_type)
            else:
                matrix_block = ops.zeros(matrix_block_shape, dtype=data_type)

            current_block_row.append(matrix_block)
        concatenated_block_rows.append(ops.concatenate(current_block_row, axis=1))

    final_matrix = ops.concatenate(concatenated_block_rows, axis=0)
    return final_matrix


def _calculate_batch_sizes(input_tensors, selected_sample_indices):
    """Calculates the batch size for each tensor."""
    if selected_sample_indices is not None:
        return [ops.shape(indices)[0] for indices in selected_sample_indices]
    return [ops.shape(tensor)[0] for tensor in input_tensors]


def _get_or_create_attention_bias(batch_sizes, input_tensors, cache):
    """
    Retrieves the attention bias from the cache or creates and caches it if not present.
    """
    tensor_shapes_cache_key = tuple(
        (batch, ops.shape(tensor)[1]) for batch, tensor in zip(batch_sizes, input_tensors)
    )

    if tensor_shapes_cache_key not in cache:
        sequence_lengths = []
        for batch_size, tensor in zip(batch_sizes, input_tensors):
            sequence_length = ops.shape(tensor)[1]
            sequence_lengths.extend([sequence_length] * batch_size)

        attention_bias_matrix = create_block_diagonal_matrix(sequence_lengths, input_tensors[0].dtype)
        cache[tensor_shapes_cache_key] = attention_bias_matrix

    return cache[tensor_shapes_cache_key]


def _concatenate_tensors(input_tensors, selected_sample_indices):
    """
    Concatenates tensors, either by selecting specific indices or combining all.
    """
    if selected_sample_indices is not None:
        flattened_tensors = [ops.reshape(tensor, (ops.shape(tensor)[0], -1)) for tensor in input_tensors]
        selected_and_concatenated_tensor = select_and_concatenate_tensors_by_indices(
            flattened_tensors, selected_sample_indices
        )
        return ops.reshape(
            selected_and_concatenated_tensor,
            (1, -1, ops.shape(input_tensors[0])[-1]),
        )
    else:
        reshaped_tensors = [ops.reshape(tensor, (1, -1, ops.shape(tensor)[-1])) for tensor in input_tensors]
        return ops.concatenate(reshaped_tensors, axis=1)


def generate_attention_bias_and_concatenate_tensors(input_tensors, selected_sample_indices=None):
    """Prepares tensors for a single attention operation.

    Creates a cached, block-diagonal attention bias to prevent cross-attention
    and concatenates the input tensors into a single tensor. Tensors can be
    concatenated fully or by selecting specific samples via indices.

    Args:
        input_tensors (list): A list of tensors to process.
        selected_sample_indices (list, optional): Indices of samples to select
            from each tensor before concatenation. If None, all samples are used.

    Returns:
        tuple: A tuple containing the attention bias tensor and the final
            concatenated tensor.
    """
    batch_sizes = _calculate_batch_sizes(input_tensors, selected_sample_indices)

    attention_bias = _get_or_create_attention_bias(batch_sizes, input_tensors, attention_bias_cache)

    concatenated_tensors = _concatenate_tensors(input_tensors, selected_sample_indices)

    return attention_bias, concatenated_tensors


def _get_stochastic_depth_parameters(input_tensors, drop_ratio):
    """Determines which samples survive stochastic depth and gets their scale factors."""
    stochastic_depth_outputs = [
        get_stochastic_depth_inputs(tensor, sample_drop_ratio=drop_ratio) for tensor in input_tensors
    ]
    surviving_sample_indices = [output[0] for output in stochastic_depth_outputs]
    scale_factors = [output[1] for output in stochastic_depth_outputs]
    return surviving_sample_indices, scale_factors


def _split_and_reshape_residuals(concatenated_residuals, input_tensors, surviving_sample_indices):
    """Splits the concatenated residual tensor back into a list of tensors."""
    split_lengths = []
    target_shapes = []
    for tensor, indices in zip(input_tensors, surviving_sample_indices):
        num_surviving_samples = ops.shape(indices)[0]
        seq_len = ops.shape(tensor)[1]
        dim = ops.shape(tensor)[2]
        split_lengths.append(num_surviving_samples * seq_len)
        target_shapes.append((num_surviving_samples, seq_len, dim))

    flat_residuals = ops.reshape(concatenated_residuals, (-1, ops.shape(concatenated_residuals)[-1]))
    split_flat_residuals = ops.split(flat_residuals, split_lengths, axis=0)

    return [ops.reshape(residual, shape) for residual, shape in zip(split_flat_residuals, target_shapes)]


def _apply_residuals_to_list(
    input_tensors,
    surviving_indices,
    split_residuals,
    scale_factors,
    scaling_vector,
):
    """Adds the calculated residuals back to the original tensors."""
    outputs = []
    for tensor, indices, residual, scale in zip(
        input_tensors, surviving_indices, split_residuals, scale_factors
    ):
        output = add_residual(tensor, indices, residual, scale, scaling_vector)
        outputs.append(output)
    return outputs


def apply_stochastic_depth_with_residual_connection(
    input_tensors,
    apply_residual,
    stochastic_depth_drop_ratio=0.0,
    residual_scaling_vector=None,
):
    """Applies a residual function with stochastic depth to a list of tensors.

    This function implements stochastic depth, a regularization technique that
    randomly drops a subset of samples (rows) from each tensor before applying
    a residual function. The surviving samples are processed, and the resulting
    residual is added back to the original tensors, scaled appropriately.

    Args:
        input_tensors (list): A list of 3D tensors to which the operation
            will be applied.
        apply_residual (callable): A function that takes a tensor and an
            attention bias (`attn_bias`) and returns a residual tensor of the
            same shape.
        stochastic_depth_drop_ratio (float, optional): The probability of
            dropping a sample from each tensor. Defaults to 0.0 (no dropping).
        residual_scaling_vector (tensor, optional): An optional scaling vector
            to be applied during the final residual addition. Defaults to None.

    Returns:
        list: A list of tensors with the same shapes as the `input_tensors`,
              containing the result of the drop, add, and residual operation.
    """
    surviving_indices, scale_factors = _get_stochastic_depth_parameters(
        input_tensors, stochastic_depth_drop_ratio
    )

    attention_bias, concatenated_tensors = generate_attention_bias_and_concatenate_tensors(
        input_tensors, surviving_indices
    )
    concatenated_residuals = apply_residual(concatenated_tensors, attn_bias=attention_bias)

    split_residuals = _split_and_reshape_residuals(concatenated_residuals, input_tensors, surviving_indices)
    final_output_tensors = _apply_residuals_to_list(
        input_tensors,
        surviving_indices,
        split_residuals,
        scale_factors,
        residual_scaling_vector,
    )
    return final_output_tensors


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
        self.norm1 = norm_layer(axis=-1, name="norm1")
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            projection_bias=proj_bias,
            attention_drop_rate=attn_drop,
            projection_drop_rate=drop,
            name="attn",
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, name="ls1") if init_values else keras.layers.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path, name="drop_path1") if drop_path > 0.0 else keras.layers.Identity()
        )

        self.norm2 = norm_layer(axis=-1, name="norm2")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, name="ls2") if init_values else keras.layers.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path, name="drop_path2") if drop_path > 0.0 else keras.layers.Identity()
        )

        self.sample_drop_ratio = drop_path

    def call(self, x, training=None):
        def apply_attention_residual(tensor):
            tensor = self.norm1(tensor)
            tensor = self.attn(tensor, training=training)
            tensor = self.ls1(tensor, training=training)
            return tensor

        def apply_feedforward_network_residual(tensor):
            tensor = self.norm2(tensor)
            tensor = self.mlp(tensor, training=training)
            tensor = self.ls2(tensor, training=training)
            return tensor

        if training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(apply_attention_residual(x), training=training)
            x = x + self.drop_path2(apply_feedforward_network_residual(x), training=training)
        else:
            x = x + apply_attention_residual(x)
            x = x + apply_feedforward_network_residual(x)

        return x


class NestedTensorBlock(Block):
    def _deconcatenate_tensors(self, processed_tensor, original_list):
        """Reshapes a processed tensor back into a list of tensors with original shapes."""
        split_lengths = [ops.shape(t)[0] * ops.shape(t)[1] for t in original_list]
        processed_flat = ops.reshape(processed_tensor, (-1, ops.shape(processed_tensor)[-1]))
        split_flat = ops.split(processed_flat, split_lengths, axis=0)
        original_shapes = [ops.shape(t) for t in original_list]
        return [
            ops.reshape(tensor, shape) for tensor, shape in zip(split_flat, original_shapes)
        ]

    def call_nested(self, x_list, training=None):
        output_list = None
        if training and self.sample_drop_ratio > 0.0:
            def apply_attention_residual(x, attention_bias=None):
                return self.attn(self.norm1(x), attention_bias=attention_bias)
            def apply_feedforward_network_residual(x, attention_bias=None):
                return self.mlp(self.norm2(x))
            processed_list = apply_stochastic_depth_with_residual_connection(
                x_list,
                apply_attention_residual,
                self.sample_drop_ratio,
                self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            processed_list = apply_stochastic_depth_with_residual_connection(
                processed_list,
                apply_feedforward_network_residual,
                self.sample_drop_ratio,
                self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
            )
            output_list = processed_list

        else:
            def apply_attention_residual(x, attention_bias=None):
                return self.ls1(self.attn(self.norm1(x), attention_bias=attention_bias))

            def apply_feedforward_network_residual(x, attention_bias=None):
                return self.ls2(self.mlp(self.norm2(x)))
            attention_mask, x_concatenated = generate_attention_bias_and_concatenate_tensors(x_list)

            x_concatenated = x_concatenated + apply_attention_residual(x_concatenated, attention_bias=attention_mask)
            x_concatenated = x_concatenated + apply_feedforward_network_residual(x_concatenated)

            output_list = self._deconcatenate_tensors(x_concatenated, x_list)

        return output_list

    def call(self, x_or_x_list, training=None):
        result = None
        if isinstance(x_or_x_list, list):
            result = self.call_nested(x_or_x_list, training=training)
        elif hasattr(x_or_x_list, "shape") and hasattr(x_or_x_list, "dtype"):
            result = super().call(x_or_x_list, training=training)
        else:
            raise TypeError(
                f"Input must be a tensor or a list of tensors, but got {type(x_or_x_list)}"
            )
        return result
