import numpy as np



def standardize(data, mean, scale):
    """Standardize the data.

    # Arguments
        data: nxd matrix to normalize
        mean: Array of means
        scale: standard deviation

    # Returns
        standardized poses2D
    # """
    return np.divide((data - mean), scale)


def destandardize(data, mean, scale):
    """Destandardize the data.

    # Arguments
        data: nxd matrix to unnormalize
        mean: Array of means
        scale: standard deviation

    # Returns
        destandardized poses3D
    # """
    return (data * scale) + mean


def filter_keypoints(keypoints, valid_args):
    """filter keypoints.

    # Arguments
        keypoints: Nx96 points in camera coordinates
        valid_args: Array of joints indices

    # Returns
        filtered keypoints
    # """
    return keypoints[:, valid_args, :]
