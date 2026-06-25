import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment


def compute_heatmaps_and_tags(outputs, num_keypoints):
    outputs = [np.transpose(np.asarray(output), [0, 3, 1, 2]) for output in outputs]  # fmt: skip
    outputs = scale_non_final_outputs(outputs, 2)
    heatmaps = average_heatmaps(outputs, num_keypoints)
    tags = outputs[0][:, num_keypoints:]
    height, width = heatmaps.shape[2:]
    heatmaps = 2.0 * resize_channels(heatmaps, 2 * width, 2 * height)
    tags = resize_channels(tags, 2 * width, 2 * height)
    return heatmaps, np.expand_dims(tags, axis=-1)


def scale_non_final_outputs(outputs, scale):
    scaled = []
    for arg, output in enumerate(outputs):
        if arg != len(outputs) - 1:
            height, width = output.shape[2:]
            output = resize_channels(output, scale * width, scale * height)
        scaled.append(output)
    return scaled


def average_heatmaps(outputs, num_keypoints):
    total = sum(output[:, :num_keypoints] for output in outputs)
    return total / len(outputs)


def resize_channels(array, width, height):
    channels = [cv2.resize(channel, (width, height)) for channel in array[0]]
    return np.stack(channels)[None]


def top_k_detections(heatmaps, tags, k):
    heatmaps = filter_peaks(heatmaps)
    num_keypoints, height, width = heatmaps.shape[1:4]
    heatmaps = heatmaps.reshape(num_keypoints, height * width)
    tags = tags.reshape(num_keypoints, height * width, -1)
    detections = []
    for keypoint_arg in range(num_keypoints):
        indices = np.argsort(heatmaps[keypoint_arg])[-k:]
        values = heatmaps[keypoint_arg][indices][:, None]
        locations = np.stack([indices % width, indices // width], axis=1)
        keypoint_tags = tags[keypoint_arg][indices]
        detections.append(np.concatenate([locations, values, keypoint_tags], 1))
    return np.stack(detections)


def filter_peaks(heatmaps):
    peaks = []
    for heatmap in heatmaps[0]:
        maxima = maximum_filter(heatmap, size=3, mode="constant")
        peaks.append(heatmap * (heatmap == maxima))
    return np.stack(peaks)[None]


def group_by_tag(detections, keypoint_order, tag_thresh, detection_thresh):
    num_keypoints, dims = detections.shape[0], detections.shape[-1]
    default = np.zeros((num_keypoints, dims))
    keypoint_dict, tag_dict = {}, {}
    for order_arg, keypoint_arg in enumerate(keypoint_order):
        keypoints = detections[keypoint_arg]
        candidates = valid_detections(keypoints, detection_thresh)
        tags = candidates[:, 3:]
        if order_arg == 0 or len(keypoint_dict) == 0:
            add_new_groups(candidates, tags, keypoint_arg, default,
                           keypoint_dict, tag_dict)
            continue
        assign_candidates(candidates, tags, keypoint_arg, default, tag_thresh,
                          keypoint_dict, tag_dict)
    return np.array(list(keypoint_dict.values()))


def valid_detections(detections, detection_thresh):
    return detections[detections[:, 2] > detection_thresh]


def add_new_groups(candidates, tags, keypoint_arg, default, keypoint_dict,
                   tag_dict):
    for tag, candidate in zip(tags, candidates):
        key = tag[0]
        keypoint_dict.setdefault(key, default.copy())[keypoint_arg] = candidate
        tag_dict[key] = [tag]


def assign_candidates(candidates, tags, keypoint_arg, default, tag_thresh,
                      keypoint_dict, tag_dict):
    keys = list(keypoint_dict.keys())
    grouped_tags = np.array([np.mean(tag_dict[key], axis=0) for key in keys])
    difference = tags[:, None, :] - grouped_tags[None, :, :]
    cost = pad_to_square(np.linalg.norm(difference, ord=2, axis=2), 1e10)
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        matched = col < len(keys) and cost[row, col] < tag_thresh
        if matched and row < len(candidates):
            key = keys[col]
            keypoint_dict[key][keypoint_arg] = candidates[row]
            tag_dict[key].append(tags[row])
        elif row < len(candidates):
            key = tags[row][0]
            keypoint_dict.setdefault(key, default.copy())
            keypoint_dict[key][keypoint_arg] = candidates[row]
            tag_dict[key] = [tags[row]]


def pad_to_square(matrix, value):
    height, width = matrix.shape
    if height > width:
        return np.pad(matrix, ((0, 0), (0, height - width)), constant_values=value)  # fmt: skip
    return np.pad(matrix, ((0, width - height), (0, 0)), constant_values=value)


def adjust_keypoints(heatmaps, grouped_keypoints):
    for person in grouped_keypoints:
        for keypoint_arg, keypoint in enumerate(person):
            if keypoint[2] <= 0:
                continue
            heatmap = heatmaps[keypoint_arg]
            y, x = keypoint[0:2]
            y = compare_vertical_neighbours(x, y, heatmap)
            x = compare_horizontal_neighbours(x, y, heatmap)
            person[keypoint_arg, 0:2] = (y + 0.5, x + 0.5)
    return grouped_keypoints


def compute_scores(grouped_keypoints):
    return [person[:, 2].mean() for person in grouped_keypoints]


def refine_keypoints(heatmaps, tags, grouped_keypoints):
    for person in grouped_keypoints:
        tags_mean = compute_tags_mean(person, tags)
        updated = refine_person_keypoints(heatmaps, tags, tags_mean)
        fill_missing_keypoints(person, updated)
    return grouped_keypoints


def compute_tags_mean(person, tags):
    keypoint_tags = []
    for keypoint_arg in range(person.shape[0]):
        if person[keypoint_arg, 2] > 0:
            x, y = person[keypoint_arg][:2].astype(np.int32)
            keypoint_tags.append(tags[keypoint_arg, y, x])
    tags_mean = np.mean(keypoint_tags, axis=0)
    return np.expand_dims(tags_mean, axis=[0, 1])


def refine_person_keypoints(heatmaps, tags, tags_mean):
    updated = []
    for keypoint_arg in range(heatmaps.shape[0]):
        heatmap = heatmaps[keypoint_arg]
        normalized = normalize_with_tags(tags[keypoint_arg], tags_mean, heatmap)
        x, y = np.unravel_index(np.argmax(normalized), heatmap.shape)
        value = heatmap[x, y]
        x, y = x + 0.5, y + 0.5
        y = compare_vertical_neighbours(x, y, heatmap)
        x = compare_horizontal_neighbours(x, y, heatmap)
        updated.append((y, x, value))
    return np.array(updated)


def normalize_with_tags(tags, tags_mean, heatmap):
    distance = ((tags - tags_mean) ** 2).sum(axis=2)
    return heatmap - np.round(np.sqrt(distance))


def fill_missing_keypoints(person, updated):
    for keypoint_arg in range(len(updated)):
        present = updated[keypoint_arg, 2] > 0 and person[keypoint_arg, 2] == 0
        if present:
            person[keypoint_arg, :3] = updated[keypoint_arg, :3]
    return person


def compare_vertical_neighbours(x, y, heatmap, offset=0.25):
    int_x, int_y = int(x), int(y)
    lower_y = min(int_y + 1, heatmap.shape[1] - 1)
    upper_y = max(int_y - 1, 0)
    return y + offset if heatmap[int_x, lower_y] > heatmap[int_x, upper_y] else y - offset  # fmt: skip


def compare_horizontal_neighbours(x, y, heatmap, offset=0.25):
    int_x, int_y = int(x), int(y)
    left_x = max(0, int_x - 1)
    right_x = min(int_x + 1, heatmap.shape[0] - 1)
    return x + offset if heatmap[right_x, int_y] > heatmap[left_x, int_y] else x - offset  # fmt: skip
