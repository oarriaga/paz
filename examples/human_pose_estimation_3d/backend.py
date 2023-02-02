import numpy as np


def standardize(poses2D, mean, scale):
    """Standardize the data.

    # Arguments
        poses2D: nxd matrix to normalize
        mean: Array of means
        scale: standard deviation

    # Returns
        standardized poses2D
    # """
    return np.divide((poses2D - mean), scale)
    

def destandardize(poses3D, mean, scale):
    """Destandardize the data.

    # Arguments
        poses3D: nxd matrix to unnormalize
        mean: Array of means
        scale: standard deviation

    # Returns
        destandardized poses3D
    # """
    return (poses3D * scale) + mean
    
    
def filter_keypoints(keypoints, valid_args):
    """filter keypoints.

    # Arguments
        keypoints: Nx96 points in camera coordinates
        valid_args: Array of joints indices

    # Returns
        filtered keypoints
    # """
    return keypoints[:, valid_args, :]
