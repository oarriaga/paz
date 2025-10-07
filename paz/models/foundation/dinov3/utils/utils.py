import keras
import numpy as np
from typing import List, Tuple, Callable


def cat_keep_shapes(
    x_list: List[keras.KerasTensor],
) -> Tuple[keras.KerasTensor, List[Tuple[int, ...]], List[int]]:
    """Flattens and concatenates a list of tensors, preserving original shape info."""
    shapes = [x.shape for x in x_list]

    num_tokens = [
        np.prod(x.shape[:-1]).item() if len(x.shape) > 1 else x.shape[0] for x in x_list
    ]
    reshaped_tensors = [
        keras.ops.reshape(x, (n, x.shape[-1])) for x, n in zip(x_list, num_tokens)
    ]
    flattened = keras.ops.concatenate(reshaped_tensors, axis=0)
    return flattened, shapes, num_tokens


def uncat_with_shapes(
    flattened: keras.KerasTensor,
    shapes: List[Tuple[int, ...]],
    num_tokens: List[int],
) -> List[keras.KerasTensor]:
    """Reverses the cat_keep_shapes operation."""
    if len(num_tokens) == 1:
        outputs_splitted = [flattened]
    else:
        split_indices = np.cumsum(num_tokens)[:-1].tolist()
        outputs_splitted = keras.ops.split(flattened, split_indices, axis=0)

    new_feature_dim = flattened.shape[-1]
    shapes_adjusted = [shape[:-1] + (new_feature_dim,) for shape in shapes]

    outputs_reshaped = [
        keras.ops.reshape(o, s) for o, s in zip(outputs_splitted, shapes_adjusted)
    ]
    return outputs_reshaped
