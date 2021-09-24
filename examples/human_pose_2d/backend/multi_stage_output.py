import numpy as np


def get_heatmaps_average(output, num_joints, with_flip, indices):
    heatmaps_average = 0
    if with_flip:
        temp = output[:, :, :, :num_joints]
        heatmaps_average += np.take(temp, indices, axis=-1)
    else:
        heatmaps_average += output[:, :, :, :num_joints]
    return heatmaps_average


def calculate_offset(with_heatmap_loss, num_joints):
    if with_heatmap_loss:
        offset = num_joints
    else:
        offset = 0
    return offset


def get_tags(output, tags, offset, indices, tag_per_joint, with_flip=False):
    tags.append(output[:, :, :, offset:])
    if with_flip and tag_per_joint:
        tags[-1] = np.take(tags[-1], indices, axis=-1)
    return tags
