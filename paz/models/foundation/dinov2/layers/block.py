import keras
from keras import ops
from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import MLP
from paz.models.foundation.dinov2.layers import SwiGLUFFNFused


def get_random_subset_of_indices(batch_size, drop_ratio):
    """Selects a random subset of indices from the batch."""
    number_of_samples_to_keep_float = ops.multiply(
        ops.cast(batch_size, "float32"), (1.0 - drop_ratio)
    )

    number_of_samples_to_keep = ops.maximum(
        ops.cast(number_of_samples_to_keep_float, "int32"), 1
    )

    all_possible_indices = ops.arange(batch_size, dtype="int32")

    selected_indices = keras.random.shuffle(all_possible_indices)[
        :number_of_samples_to_keep
    ]

    return selected_indices


def prepare_and_scale_residual_tensor(
    residual_tensor_to_scale,
    number_of_samples_in_residual,
    scale_factor,
):
    """
    Flattens a residual tensor to two dimensions and scales its values.

    This function prepares the residual to be added to another tensor by
    ensuring it's in a compatible (flattened) format and scaled appropriately.

    Args:
        residual_tensor_to_scale: The tensor containing the residual values.
        number_of_samples_in_residual: The number of samples in the residual,
                                       which corresponds to the first dimension
                                       of the reshaped tensor.
        scale_factor: A float value to multiply against the residual values.

    Returns:
        A 2D tensor representing the flattened and scaled residual.
    """
    flattened_residual_tensor = ops.reshape(
        residual_tensor_to_scale, (number_of_samples_in_residual, -1)
    )

    scaled_flattened_residual_tensor = ops.multiply(
        flattened_residual_tensor, scale_factor
    )

    return scaled_flattened_residual_tensor


def selectively_add_residual_to_tensor(
    original_tensor,
    indices_for_update,
    scaled_residual_to_add,
):
    """
    Adds a scaled residual to a selected subset of an original tensor.

    This function modifies only the specified rows (samples) of the original
    tensor by adding the corresponding residual values.

    Args:
        original_tensor: The base tensor to be modified.
        indices_for_update: A 1D tensor of indices specifying which rows
                            of the original_tensor to update.
        scaled_residual_to_add: A 2D tensor containing the scaled residual
                                values to be added.

    Returns:
        A new tensor with the same shape as the original_tensor, but with the
        residual added to the specified rows.
    """
    original_tensor_shape = ops.shape(original_tensor)
    batch_size = original_tensor_shape[0]

    flattened_original_tensor = ops.reshape(original_tensor, (batch_size, -1))

    existing_values_at_update_indices = ops.take(
        flattened_original_tensor, indices_for_update, axis=0
    )

    updated_values_for_subset = ops.add(
        existing_values_at_update_indices, scaled_residual_to_add
    )

    scatter_update_indices = ops.expand_dims(indices_for_update, axis=1)

    updated_flattened_tensor = ops.scatter_update(
        flattened_original_tensor,
        scatter_update_indices,
        updated_values_for_subset,
    )

    final_tensor_in_original_shape = ops.reshape(
        updated_flattened_tensor, original_tensor_shape
    )

    return final_tensor_in_original_shape


def scale_and_add_residual(
    original_input_tensor,
    selected_sample_indices,
    residual_to_add,
    residual_scale_factor,
):
    """Scales and adds a residual to a subset of the original tensor."""
    number_of_samples = ops.shape(selected_sample_indices)[0]
    scaled_residual = prepare_and_scale_residual_tensor(
        residual_tensor_to_scale=residual_to_add,
        number_of_samples_in_residual=number_of_samples,
        scale_factor=residual_scale_factor,
    )

    final_output_tensor = selectively_add_residual_to_tensor(
        original_tensor=original_input_tensor,
        indices_for_update=selected_sample_indices,
        scaled_residual_to_add=scaled_residual,
    )

    return final_output_tensor


def drop_add_residual_stochastic_depth(
    input_tensor,
    apply_residual,
    sample_drop_ratio=0.0,
):
    """Applies a residual function to a random subset of
    a batch (Stochastic Depth).

    This function implements a form of Stochastic Depth,
    a regularization technique. Instead of applying the residual
    function to the entire input tensor, it is applied only to
    a randomly selected subset of the samples in the batch.
    The resulting residual is then scaled to preserve the expected value
    of the output before being added back to the corresponding
    original samples.
    Samples not selected for the residual computation are passed through
    unchanged (identity path).

    Args:
        input_tensor:
            The input tensor, where the first dimension is the batch size.
        apply_residual: A callable (e.g., a layer or model) that computes the
            residual transformation.
            It will be called on a subset of `input_tensor`.
        sample_drop_ratio: The probability of dropping a sample from the
            residual path. A value of 0.0 means no samples are dropped
            (standard residual connection), while a value of 1.0 would mean all
            samples are dropped.

    Returns:
        A tensor with the same shape as the input `input_tensor`,
        where the scaled residual has been added to a random
        subset of the batch samples.
    """
    batch_size = ops.shape(input_tensor)[0]

    selected_sample_indices = get_random_subset_of_indices(
        batch_size, sample_drop_ratio
    )

    input_subset = ops.take(input_tensor, selected_sample_indices, axis=0)
    residual_from_subset = apply_residual(input_subset)

    batch_size_as_float = ops.cast(batch_size, input_tensor.dtype)
    number_of_selected_samples_as_float = ops.cast(
        ops.shape(selected_sample_indices)[0], input_tensor.dtype
    )
    residual_scale_factor = ops.divide(
        batch_size_as_float, number_of_selected_samples_as_float
    )
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
        x: The input tensor. The batch size is inferred from its
           first dimension.
        sample_drop_ratio: The probability of dropping
            a sample from processing.
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


def add_residual(
    x, selected_sample_indices, residual, residual_scale_factor, scaling_vector=None
):
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
        final_residual = ops.multiply(
            element_wise_scaled_residual, residual_scale_factor
        )
    else:
        final_residual = ops.multiply(residual, residual_scale_factor)

    existing_values = ops.take(x, selected_sample_indices, axis=0)
    updated_values = ops.add(existing_values, final_residual)
    scatter_indices = ops.expand_dimensions(selected_sample_indices, axis=1)
    x_plus_residual = ops.scatter_update(x, scatter_indices, updated_values)

    return x_plus_residual


attention_bias_cache = {}


def select_and_concatenate_tensors_by_indices(
    list_of_tensors, list_of_indices_per_tensor
):
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

    for current_tensor, current_indices in zip(
        list_of_tensors, list_of_indices_per_tensor
    ):
        selected_slice = ops.take(current_tensor, current_indices, axis=0)

        selected_tensor_slices.append(selected_slice)

    final_concatenated_tensor = ops.concatenate(selected_tensor_slices, axis=0)

    return final_concatenated_tensor


def create_block_diagonal_matrix(block_sizes, data_type):
    """Constructs a block diagonal matrix from a list of block sizes.

    This function creates a larger matrix composed of smaller square
    matrices (blocks) placed along the main diagonal.
    Each block on the diagonal is filled with ones, and all other
    off-diagonal blocks are filled with zeros.

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


def calculate_batch_sizes(input_tensors, selected_sample_indices):
    """Calculates the batch size for each tensor."""
    batch_sizes = []
    if selected_sample_indices is not None:
        for indices in selected_sample_indices:
            batch_size = ops.shape(indices)[0]
            batch_sizes.append(batch_size)

    else:
        for tensor in input_tensors:
            batch_size = ops.shape(tensor)[0]
            batch_sizes.append(batch_size)
    return batch_sizes


def get_or_create_attention_bias(batch_sizes, input_tensors, cache):
    """
    Retrieves the attention bias from the cache or creates
    and caches it if not present.
    """
    cache_key_list = []
    for batch, tensor in zip(batch_sizes, input_tensors):
        shape_tuple = (batch, ops.shape(tensor)[1])
        cache_key_list.append(shape_tuple)

    tensor_shapes_cache_key = tuple(cache_key_list)

    if tensor_shapes_cache_key not in cache:
        sequence_lengths = []
        for batch_size, tensor in zip(batch_sizes, input_tensors):
            sequence_length = ops.shape(tensor)[1]
            sequence_lengths.extend([sequence_length] * batch_size)

        attention_bias_matrix = create_block_diagonal_matrix(
            sequence_lengths, input_tensors[0].dtype
        )
        cache[tensor_shapes_cache_key] = attention_bias_matrix

    return cache[tensor_shapes_cache_key]


def concatenate_tensors(input_tensors, selected_sample_indices):
    """
    Concatenates tensors, by selecting specific indices or combining all.
    """
    if selected_sample_indices is not None:
        flattened_tensors = [
            ops.reshape(tensor, (ops.shape(tensor)[0], -1)) for tensor in input_tensors
        ]
        selected_and_concatenated_tensor = select_and_concatenate_tensors_by_indices(
            flattened_tensors, selected_sample_indices
        )
        concatenated_tensor = ops.reshape(
            selected_and_concatenated_tensor,
            (1, -1, ops.shape(input_tensors[0])[-1]),
        )
    else:
        reshaped_tensors = [
            ops.reshape(tensor, (1, -1, ops.shape(tensor)[-1]))
            for tensor in input_tensors
        ]
        concatenated_tensor = ops.concatenate(reshaped_tensors, axis=1)

    return concatenated_tensor


def generate_attention_bias_and_concatenate_tensors(
    input_tensors, selected_sample_indices=None
):
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
    batch_sizes = calculate_batch_sizes(input_tensors, selected_sample_indices)

    attention_bias = get_or_create_attention_bias(
        batch_sizes, input_tensors, attention_bias_cache
    )

    concatenated_tensors = concatenate_tensors(input_tensors, selected_sample_indices)

    return attention_bias, concatenated_tensors


def get_stochastic_depth_parameters(input_tensors, drop_ratio):
    """Determines which samples survive stochastic depth
    and gets their scale factors.
    """
    surviving_sample_indices = []
    scale_factors = []

    for tensor in input_tensors:
        surviving_indices, scale_factor = get_stochastic_depth_inputs(
            tensor, sample_drop_ratio=drop_ratio
        )

        surviving_sample_indices.append(surviving_indices)
        scale_factors.append(scale_factor)

    return surviving_sample_indices, scale_factors


def calculate_split_shapes(input_tensors, surviving_sample_indices):
    """Calculates the necessary split lengths and target shapes for the residuals."""
    split_lengths = []
    target_shapes = []
    for tensor, indices in zip(input_tensors, surviving_sample_indices):
        number_of_surviving_samples = ops.shape(indices)[0]
        sequence_length = ops.shape(tensor)[1]
        dimension = ops.shape(tensor)[2]

        split_lengths.append(number_of_surviving_samples * sequence_length)
        target_shapes.append((number_of_surviving_samples, sequence_length, dimension))

    return split_lengths, target_shapes


def flatten_and_split_main_residuals_tensor(split_lengths, concatenated_residuals):
    flat_residuals = ops.reshape(
        concatenated_residuals, (-1, ops.shape(concatenated_residuals)[-1])
    )
    split_flat_residuals = ops.split(flat_residuals, split_lengths, axis=0)
    return split_flat_residuals


def reshape_each_split_part_to_target_shape(split_flat_residuals, target_shapes):
    """Reshapes each split part of the residuals to its target shape."""
    reshaped_residuals = []
    for residual, shape in zip(split_flat_residuals, target_shapes):
        reshaped_tensor = ops.reshape(residual, shape)
        reshaped_residuals.append(reshaped_tensor)
    return reshaped_residuals


def split_and_reshape_residuals(
    concatenated_residuals, input_tensors, surviving_sample_indices
):
    """Splits the concatenated residual tensor back into a list of tensors."""
    split_lengths, target_shapes = calculate_split_shapes(
        input_tensors, surviving_sample_indices
    )

    split_flat_residuals = flatten_and_split_main_residuals_tensor(
        split_lengths, concatenated_residuals
    )

    reshaped_residuals = reshape_each_split_part_to_target_shape(
        split_flat_residuals, target_shapes
    )

    return reshaped_residuals


def apply_residuals_to_list(
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
            attention bias (`attention_bias`) and returns a residual tensor of the
            same shape.
        stochastic_depth_drop_ratio (float, optional): The probability of
            dropping a sample from each tensor. Defaults to 0.0 (no dropping).
        residual_scaling_vector (tensor, optional): An optional scaling vector
            to be applied during the final residual addition. Defaults to None.

    Returns:
        list: A list of tensors with the same shapes as the `input_tensors`,
              containing the result of the drop, add, and residual operation.
    """
    surviving_indices, scale_factors = get_stochastic_depth_parameters(
        input_tensors, stochastic_depth_drop_ratio
    )

    attention_bias, concatenated_tensors = (
        generate_attention_bias_and_concatenate_tensors(
            input_tensors, surviving_indices
        )
    )
    concatenated_residuals = apply_residual(
        concatenated_tensors, attention_bias=attention_bias
    )

    split_residuals = split_and_reshape_residuals(
        concatenated_residuals, input_tensors, surviving_indices
    )
    final_output_tensors = apply_residuals_to_list(
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
        dimension,
        number_of_heads,
        mlp_ratio=4.0,
        use_query_key_value_bias=False,
        use_projection_bias=True,
        use_feedforward_network_bias=True,
        drop_rate=0.0,
        attention_drop_rate=0.0,
        init_values=None,
        drop_path=0.0,
        activation_layer=keras.activations.gelu,
        normalization_layer=keras.layers.LayerNormalization,
        attention_class=Attention,
        feedforward_network_layer="mlp",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.normalization1 = normalization_layer(axis=-1, name="norm1")
        self.attention = attention_class(
            dimension,
            number_of_heads=number_of_heads,
            use_query_key_value_bias=use_query_key_value_bias,
            use_projection_bias=use_projection_bias,
            attention_drop_rate=attention_drop_rate,
            projection_drop_rate=drop_rate,
        )
        self.layer_scale_1 = (
            LayerScale(dimension, init_values=init_values, name="ls1")
            if init_values
            else keras.layers.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path, name="drop_path1")
            if drop_path > 0.0
            else keras.layers.Identity()
        )

        self.normalization2 = normalization_layer(axis=-1, name="norm2")
        mlp_hidden_dimension = int(dimension * mlp_ratio)
        if feedforward_network_layer == "mlp":
            feedforward_network_layer_class = MLP
        elif feedforward_network_layer in ["swiglu", "swiglufused"]:
            feedforward_network_layer_class = SwiGLUFFNFused
        else:
            raise ValueError(f"Unknown FFN layer: {feedforward_network_layer}")

        mlp_hidden_dimension = int(dimension * mlp_ratio)
        self.mlp = feedforward_network_layer_class(
            input_features=dimension,
            hidden_features=mlp_hidden_dimension,
            activation_layer=activation_layer,
            drop_rate=drop_rate,
            use_bias=use_feedforward_network_bias,
        )
        self.layer_scale_2 = (
            LayerScale(dimension, init_values=init_values, name="ls2")
            if init_values
            else keras.layers.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path, name="drop_path2")
            if drop_path > 0.0
            else keras.layers.Identity()
        )

        self.sample_drop_ratio = drop_path

    def call(self, x, training=None):
        normalized_x_1 = self.normalization1(x)
        attention_output = self.attention(normalized_x_1, training=training)
        scaled_attention = self.layer_scale_1(attention_output, training=training)

        if training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(scaled_attention, training=training)
        else:
            x = x + scaled_attention

        normalized_x_2 = self.normalization2(x)
        mlp_output = self.mlp(normalized_x_2, training=training)
        scaled_mlp = self.layer_scale_2(mlp_output, training=training)

        if training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path2(scaled_mlp, training=training)
        else:
            x = x + scaled_mlp

        return x


class NestedTensorBlock(Block):

    def apply_attention_residual(self, x, attention_bias=None):
        normalized_x = self.normalization1(x)
        attention_output = self.attention(normalized_x, attention_bias=attention_bias)

        return attention_output

    def apply_feedforward_network_residual(self, x):
        normalized_x = self.normalization2(x)
        mlp_output = self.mlp(normalized_x)

        return mlp_output

    def apply_scaled_attention_residual(self, x, attention_bias=None):
        attention_output = self.apply_attention_residual(
            x, attention_bias=attention_bias
        )
        scaled_output = self.layer_scale_1(attention_output)
        return scaled_output

    def apply_scaled_feedforward_network_residual(self, x):
        mlp_output = self.apply_feedforward_network_residual(x)
        scaled_output = self.layer_scale_2(mlp_output)
        return scaled_output

    def deconcatenate_tensors(self, processed_tensor, original_list):
        """Reshapes a processed tensor back into a list of tensors
        with original shapes."""
        split_lengths = []
        original_shapes = []

        for t in original_list:
            shape = ops.shape(t)
            original_shapes.append(shape)
            split_lengths.append(shape[0] * shape[1])

        processed_flat = ops.reshape(
            processed_tensor, (-1, ops.shape(processed_tensor)[-1])
        )
        split_flat = ops.split(processed_flat, split_lengths, axis=0)

        result_list = []
        for tensor, shape in zip(split_flat, original_shapes):
            reshaped_tensor = ops.reshape(tensor, shape)
            result_list.append(reshaped_tensor)

        return result_list

    def call_nested(self, x_list, training=None):
        output_list = None
        if training and self.sample_drop_ratio > 0.0:
            processed_list = apply_stochastic_depth_with_residual_connection(
                x_list,
                self.apply_attention_residual,
                self.sample_drop_ratio,
                (
                    self.layer_scale_1.gamma
                    if isinstance(self.layer_scale_1, LayerScale)
                    else None
                ),
            )
            processed_list = apply_stochastic_depth_with_residual_connection(
                processed_list,
                self.apply_feedforward_network_residual,
                self.sample_drop_ratio,
                (
                    self.layer_scale_2.gamma
                    if isinstance(self.layer_scale_2, LayerScale)
                    else None
                ),
            )
            output_list = processed_list

        else:
            attention_mask, x_concatenated = (
                generate_attention_bias_and_concatenate_tensors(x_list)
            )

            x_concatenated = x_concatenated + self.apply_scaled_attention_residual(
                x_concatenated, attention_bias=attention_mask
            )
            x_concatenated = (
                x_concatenated
                + self.apply_scaled_feedforward_network_residual(x_concatenated)
            )

            output_list = self.deconcatenate_tensors(x_concatenated, x_list)

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
