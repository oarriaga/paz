import numpy as np
import tensorflow as tf


def append_values(dictionary, lists, keys):
    """Append dictionary values to lists

    # Arguments
        dictionary: dict
        lists: List of lists
        keys: Keys to dictionary values
    """
    if len(keys) != len(lists):
        assert ValueError('keys and lists must have same length')
    for key_arg, key in enumerate(keys):
        lists[key_arg].append(dictionary[key])
    return lists


def append_lists(intro_lists, outro_lists):
    """Appends multiple new values in intro lists to multiple outro lists

    # Arguments
        intro_lists: List of lists
        outro_lists: List of lists

    # Returns
        Lists with new values of intro lists
    """
    for intro_list, outro_list in zip(intro_lists, outro_lists):
        outro_list.append(intro_list)
    return outro_lists


def get_upper_multiple(x, multiple=64):
    """Returns the upper multiple of 'multiple' to the x.

    # Arguments
        x: Int.
        multiple: Int.

    # Returns
        upper multiple. Int.
    """
    x = x + (multiple - 1)
    floor_value = x // multiple
    upper_multiple = floor_value * multiple
    return upper_multiple


def resize_with_same_aspect_ratio(image, input_size, multiple=64):
    """Resize the sort side of the input image to input_size and keep
    the aspect ratio.

    # Arguments
        input_size: Dimension to be resized. Int.
        H: Int.
        W: Int.

    # Returns
        resized H and W.
    """
    H, W = np.sort(image.shape[:2])
    H_resized = int(input_size)
    W_resized = input_size / H
    W_resized = W_resized * W
    W_resized = int(get_upper_multiple(W_resized, multiple))
    size = np.array([W_resized, H_resized])
    return size


def get_transformation_scale(image, size, scaling_factor):
    """Caluclte scale of resized H and W.

    # Arguments
        H: Int.
        H_resized: Int.
        H_resized: Int.
        scaling_factor: Int.

    # Returns
        scaled H and W
    """
    H, W = image.shape[:2]
    H_resized, W_resized = size

    if H < W:
        H_resized, W_resized = W_resized, H_resized
    H, W = np.sort([H, W])

    scale_H = H / scaling_factor
    aspect_ratio = W_resized / H_resized
    scale_W = aspect_ratio * scale_H
    scale = np.array([scale_W, scale_H])
    return scale


def compare_vertical_neighbours(x, y, image, offset=0.25):
    """Compare two vertical neighbors and add an offset to the smaller one.

    # Arguments
        x: Int. x coordinate of pixel to be compared.
        y: Int. y coordinate of pixel to be compared.
        image: Array.
        offset: Float.
    """
    int_x, int_y = int(x), int(y)
    lower_y = min(int_y + 1, image.shape[1] - 1)
    upper_y = max(int_y - 1, 0)
    if image[int_x, lower_y] > image[int_x, upper_y]:
        y = y + offset
    else:
        y = y - offset
    return y


def compare_horizontal_neighbours(x, y, image, offset=0.25):
    """Compare two horizontal neighbors and add an offset to the smaller one.

    # Arguments
        x: Int. x coordinate of pixel to be compared.
        y: Int. y coordinate of pixel to be compared.
        image: Array.
        offset: Float.
    """
    int_x, int_y = int(x), int(y)
    left_x = max(0, int_x - 1)
    right_x = min(int_x + 1, image.shape[0] - 1)
    if image[right_x, int_y] > image[left_x, int_y]:
        x = x + offset
    else:
        x = x - offset
    return x


def get_all_indices_of_array(array):
    """Get all the indices of an array.

    # Arguments
        array: Array

    # Returns
        Array. Array with the indices of the input array
    """
    all_indices = np.ndarray(array.shape)
    all_indices.fill(True)
    all_indices = np.where(all_indices)
    all_indices = np.array(all_indices).T
    print(all_indices)
    return all_indices


def gather_nd(array, indices, axis):
    """Take the value from the input array on the given indices along the
    given axis.

    # Arguments
        array: Array
        indices: list/Array. values to be gathered from
        axis: Int. Axis along which to gather values.

    # Returns
        Array. Gathered values from the input array
    """
    gathered = np.take_along_axis(array, indices, axis=axis)
    return gathered


def calculate_norm(vector, order=None, axis=None):
    """Calculates the norm of vector.

    # Arguments
        x: List of spatial coordinates (x, y, z)
    """
    return np.linalg.norm(vector, ord=order, axis=axis)


def tensor_to_numpy(tensor):
    """Convert a tensor to a Array.

    # Arguments
        tensor: multidimensional array of type tensor
    """
    return tensor.cpu().numpy()


def pad_matrix(matrix, pool_size=(3, 3), strides=(1, 1),
               padding='valid', value=0):
    """Pad an array

    # Arguments
        matrix: Array.
        padding: String. Type of padding
        value: Int. Value to be added in the padded area.
        poolsize: Int. How many rows and colums to be padded for 'same' padding
    """
    matrix = np.array(matrix)
    H, W = matrix.shape[:2]
    if padding == 'valid':
        padding = ((0, 0), (0, 0))
    if padding == 'square':
        if H > W:
            padding = ((0, 0), (0, H - W))
        else:
            padding = ((0, W - H), (0, 0))
    if padding == 'same':
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if H % strides[0] == 0:
            height_pad = np.max((pool_size[0] - strides[0]), 0)
        else:
            height_pad = np.max(pool_size[0] - (H % strides[0]), 0)
        if W % strides[1] == 0:
            width_pad = np.max((pool_size[1] - strides[1]), 0)
        else:
            width_pad = np.max(pool_size[1] - (W % strides[1]), 0)

        pad_top = height_pad // 2
        pad_bottom = height_pad - pad_top
        pad_left = width_pad // 2
        pad_right = width_pad - pad_left
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    return np.pad(matrix, padding, mode='constant', constant_values=value)


def max_pooling_2d(image, pool_size=3, strides=1, padding='same'):
    """Returns the maximum pooled value of an image.

    # Arguments
        image: Array.
        poolsize: Int or list of len 2. Window size for each pool
        padding: String. Type of padding
    """
    if not isinstance(strides, int):
        strides = strides[0]
    if not isinstance(pool_size, int):
        pool_size = pool_size[0]

    if padding == 'valid':
        max_image = np.zeros((image.shape[0] - pool_size + 1,
                              image.shape[1] - pool_size + 1))
    if padding == 'same':
        max_image = np.zeros_like(image)

    image = pad_matrix(image, pool_size, strides, padding)
    H, W = image.shape[:2]
    for y in range(0, H - pool_size + 1, strides):
        for x in range(0, W - pool_size + 1, strides):
            max_image[y][x] = np.max(image[y:y + pool_size, x:x + pool_size])
    return max_image


def predict(x, model, preprocess=None, postprocess=None):
    """Preprocess, predict and postprocess input.
    # Arguments
        x: Input to model
        model: Callable i.e. Keras model.
        preprocess: Callable, used for preprocessing input x.
        postprocess: Callable, used for postprocessing output of model.

    # Note
        If model outputs a tf.Tensor is converted automatically to numpy array.
    """
    if preprocess is not None:
        x = preprocess(x)
    y = model(x)
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    if postprocess is not None:
        y = postprocess(y)
    return y
