import numpy as np


def get_heatmap_sum(output, num_joints, heatmap_sum):
    heatmap_sum = heatmap_sum + output[:, :num_joints, :, :]
    return heatmap_sum


def get_heatmap_sum_with_flip(output, num_joints, indices, heatmap_sum):
    output = np.flip(output, [3])
    heatmaps = output[:, :num_joints, :, :]
    heatmap_sum = heatmap_sum + np.take(heatmaps, indices, axis=1)
    return heatmap_sum


def get_tags(output, num_joint):
    tags = output[:, num_joint:, :, :]
    return tags


def get_tags_with_flip(output, num_joint, indices):
    output = np.flip(output, [3])
    tags = output[:, num_joint:, :, :]
    tags = np.take(tags, indices, axis=1)
    return tags
