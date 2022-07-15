import numpy as np


def get_keypoints_heatmap(heatmaps, num_keypoints, indices=None, axis=1):
    """Extract the heatmaps that only contains the keypoints.

    # Arguments
        heatmaps: Numpy array of shape (1, 2*num_keypoints, H, W)
        num_keypoints: Int.
        indices: List. Indices of the heatmaps to extract.
        axis: Int.

    # Returns
        keypoints: Numpy array of shape (1, num_keypoints, H, W)
    """
    keypoints = np.take(heatmaps, np.arange(num_keypoints), axis)
    if indices is not None:
        keypoints = np.take(keypoints, indices, axis)
    return keypoints


def get_tags_heatmap(heatmaps, num_keypoints, indices=None, axis=1):
    """Extract the heatmaps that only contains the tags.

    # Arguments
        heatmaps: Numpy array of shape (1, 2*num_keypoints, H, W)
        num_keypoints: Int.
        indices: List. Indices of the heatmaps to extract.
        axis: Int.

    # Returns
        tags: Numpy array of shape (1, num_keypoints, H, W)
    """
    n = heatmaps.shape[axis]
    tags = np.take(heatmaps, np.arange(num_keypoints, n), axis)
    if indices is not None:
        tags = np.take(tags, indices, axis)
    return tags


def get_keypoints_locations(indices, image_width):
    """Calculate the location of keypoints in an image.

    # Arguments
        indices: Numpy array. Indices of the keypoints in the heatmap.
        Image width: Int.

    # Returns
        coordinate: Numpy array. locations of keypoints
    """
    x = (indices % image_width).astype(np.int64)
    y = (indices / image_width).astype(np.int64)
    coordinates = np.stack((x, y), axis=3)
    return np.squeeze(coordinates)


def get_top_k_keypoints_numpy(heatmaps, k):
    """Numpy implementation of get_top_k_keypoints from heatmaps.

    # Arguments
        heatmaps: Keypoints heatmaps. Numpy array of shape
                  (1, num_keypoints, H, W)
        k: Int. Maximum number of instances to return.

    # Returns
        values: Numpy array. Value of heatmaps at top k keypoints
        indices: Numpy array. Indices of top k keypoints.
    """
    num_of_objects, num_of_keypoints = heatmaps.shape[:2]
    indices = np.zeros((num_of_objects, num_of_keypoints, k), dtype=np.int)
    values = np.zeros((num_of_objects, num_of_keypoints, k))
    for object_arg in range(num_of_objects):
        for keypoint_arg in range(num_of_keypoints):
            top_k_indices = np.argsort(heatmaps[object_arg][keypoint_arg])[-k:]
            top_k_values = heatmaps[object_arg][keypoint_arg][top_k_indices]
            indices[object_arg][keypoint_arg] = top_k_indices
            values[object_arg][keypoint_arg] = top_k_values
    return np.squeeze(values), indices


def get_valid_detections(detection, detection_thresh):
    """Accept the keypoints whose score is greater than the
       detection threshold.

    # Arguments
        detection: Numpy array. Contains the location, value, and
        tags of the keypoints
        detection_thresh: Float. Detection threshold for the keypoint
    """
    mask = detection[:, 2] > detection_thresh
    valid_detection = detection[mask]
    return valid_detection
