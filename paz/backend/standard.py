import numpy as np


# def get_upper_multiple(x, multiple=64):
#     """Returns the upper multiple of 'multiple' to the x.

#     # Arguments
#         x: Int.
#         multiple: Int.

#     # Returns
#         upper multiple. Int.
#     """
#     x = x + (multiple - 1)
#     floor_value = x // multiple
#     upper_multiple = floor_value * multiple
#     return upper_multiple


# def resize_with_same_aspect_ratio(image, input_size, multiple=64):
#     '''
#     Resize the sort side of the input image to input_size and keep
#     the aspect ratio.

#     # Arguments
#         input_size: Dimension to be resized. Int.
#         H: Int.
#         W: Int.

#     # Returns
#         resized H and W.
#     '''
#     H, W = np.sort(image.shape[:2])
#     H_resized = int(input_size)
#     W_resized = input_size / H
#     W_resized = W_resized * W
#     W_resized = int(get_upper_multiple(W_resized, multiple))
#     size = np.array([W_resized, H_resized])
#     return size


# def get_transformation_scale(image, size, scaling_factor):
#     '''
#     Caluclte scale of resized H and W.

#     # Arguments
#         H: Int.
#         H_resized: Int.
#         H_resized: Int.
#         scaling_factor: Int.

#     # Returns
#         scaled H and W
#     '''
#     H, W = image.shape[:2]
#     H_resized, W_resized = size

#     if H < W:
#         H_resized, W_resized = W_resized, H_resized
#     H, W = np.sort([H, W])

#     scale_H = H / scaling_factor
#     aspect_ratio = W_resized / H_resized
#     scale_W = aspect_ratio * scale_H
#     scale = np.array([scale_W, scale_H])
#     return scale



def get_dims_x64(dims, multiple=64):
    dims = dims + (multiple - 1)
    floor_value = dims // multiple
    dims = floor_value * multiple
    return dims


def get_transformation_size(input_size, dims1, dims2):
    '''
    Resize the short side of the input image to 512 and keep
    the aspect ratio.
    '''
    dims1_resized = int(input_size)
    dims2_resized = input_size / dims1
    dims2_resized = dims2_resized * dims2
    dims2_resized = int(get_dims_x64(dims2_resized, 64))
    return dims1_resized, dims2_resized


def get_transformation_scale(dims1, dims1_resized, dims2_resized,
                             scaling_factor):
    scale_dims1 = dims1 / scaling_factor
    scale_dims2 = dims2_resized / dims1_resized
    scale_dims2 = scale_dims2 * dims1
    scale_dims2 = scale_dims2 / scaling_factor
    return scale_dims1, scale_dims2





def add_offset(x, offset=0.25):
    x = x + offset
    return x


def compare_vertical_neighbours(x, y, image, offset=0.25):
    int_x, int_y = int(x), int(y)
    lower_y = min(int_y + 1, image.shape[1] - 1)
    upper_y = max(int_y - 1, 0)
    if image[int_x, lower_y] > image[int_x, upper_y]:
        y = add_offset(y, offset)
    else:
        y = add_offset(y, -1*offset)
    return y


def compare_horizontal_neighbours(x, y, image, offset=0.25):
    int_x, int_y = int(x), int(y)
    left_x = max(0, int_x - 1)
    right_x = min(int_x + 1, image.shape[0] - 1)
    if image[right_x, int_y] > image[left_x, int_y]:
        x = add_offset(x, offset)
    else:
        x = add_offset(x, -1*offset)
    return x


def top_k_heatmaps(heatmaps, k):
    max_= []
    val_ = []
    a, b, c = heatmaps.shape
    for i in range(a):
        max = []
        val = []
        for j in range(b):
            indices = np.argsort(heatmaps[i][j])[-k:]
            max.append(indices)
            val.append(heatmaps[i][j][indices])
        max_.append(max)
        val_.append(val)
    return np.array(max_), np.array(val_)
