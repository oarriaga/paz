import numpy as np
import tensorflow as tf
from munkres import Munkres
import cv2


JOINT_ORDER = [i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12,
                             13, 8, 9, 10, 11, 14, 15, 16, 17]]
NUM_JOINTS = len(JOINT_ORDER)
DETECTION_THRESH = 0.2
MAX_NUM_PEOPLE = 30
IGNORE_TOO_MUCH = False
USE_DETECTION_VAL = True
TAG_THRESH = 1
TAG_PER_JOINT = True


def update_dictionary(tags, joints, idx, default, joint_dict, tag_dict):
    for tag, joint in zip(tags, joints):
        key = tag[0]
        joint_dict.setdefault(key, np.copy(default))[idx] = joint
        tag_dict[key] = [tag]


def group_keys_and_tags(joint_dict, tag_dict):
    grouped_keys = list(joint_dict.keys())[:MAX_NUM_PEOPLE]
    grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]
    return grouped_keys, grouped_tags


def calculate_norm(joints, grouped_tags, order=2):
    difference = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
    norm = np.linalg.norm(difference, ord=order, axis=2)
    return difference, norm


def concatenate_zeros(metrix, shape):
    concatenated = np.concatenate((metrix, np.zeros(shape)+1e10), axis=1)
    return concatenated


def shortest_L2_distance(scores):
    munkres = Munkres()
    lowest_cost_pairs = munkres.compute(scores)
    lowest_cost_pairs = np.array(lowest_cost_pairs).astype(np.int32)
    return lowest_cost_pairs


def get_tags_and_joints(tag_k, loc_k, val_k, idx):
    tags = tag_k[idx]
    joints = np.concatenate((loc_k[idx], val_k[idx, :, None],
                            tag_k[idx]), 1)
    mask = joints[:, 2] > DETECTION_THRESH
    tags = tags[mask]
    joints = joints[mask]
    return tags, joints


def match_by_tag(input_):
    tag_k, loc_k, val_k = input_.values()
    joint_dict = {}
    tag_dict = {}
    default = np.zeros((NUM_JOINTS, tag_k.shape[2] + 3))

    for i in range(NUM_JOINTS):
        idx = JOINT_ORDER[i]
        tags, joints = get_tags_and_joints(tag_k, loc_k, val_k, idx)

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            update_dictionary(tags, joints, idx, default, joint_dict, tag_dict)

        else:
            grouped_keys, grouped_tags = group_keys_and_tags(
                                            joint_dict, tag_dict)

            if IGNORE_TOO_MUCH and len(grouped_keys) == MAX_NUM_PEOPLE:
                continue

            difference, norm = calculate_norm(joints, grouped_tags)
            norm_copy = np.copy(norm)

            num_added, num_grouped = difference.shape[:2]

            if num_added > num_grouped:
                shape = (num_added, (num_added - num_grouped))
                norm = concatenate_zeros(norm, shape)

            lowest_cost_pairs = shortest_L2_distance(norm)

            for row, col in lowest_cost_pairs:
                if (
                    row < num_added
                    and col < num_grouped
                    and norm_copy[row][col] < TAG_THRESH
                   ):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])

                else:
                    update_dictionary(tags[row], joints[row], idx, default,
                                      joint_dict, tag_dict)

    return np.array([[joint_dict[i] for i in joint_dict]]).astype(np.float32)


def non_maximum_supressions(detection_boxes):
    detection_boxes = tf.transpose(detection_boxes, [0, 2, 3, 1])
    maxm = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1,
                                        padding='same')(detection_boxes)
    maxm = tf.math.equal(maxm, detection_boxes)
    maxm = tf.cast(maxm, tf.float32)
    filtered_box = detection_boxes * maxm
    return filtered_box


def torch_gather(x, indices, gather_axis=2):
    x = tf.cast(x, tf.int64)
    indices = tf.cast(indices, tf.int64)
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    return tf.reshape(gathered, indices.shape)


def tensor_to_numpy(tensor):
    return tensor.cpu().numpy()


def top_k(boxes, tag):
    box = non_maximum_supressions(boxes)
    box = tf.transpose(box, [0, 3, 1, 2])
    num_images, num_joints = box.get_shape()[:2]
    H, W = box.get_shape()[2:4]
    box = tf.reshape(box, [num_images, num_joints, -1])

    val_k, indices = tf.math.top_k(box, MAX_NUM_PEOPLE)
    tag = tf.reshape(tag, [tag.get_shape()[0], tag.get_shape()[1], W*H, -1])
    if not TAG_PER_JOINT:
        tag = tag.expand(-1, NUM_JOINTS, -1, -1)

    tag_k = tf.stack(
        [
            torch_gather(tag[:, :, :, 0], indices)
            for i in range(tag.get_shape()[3])
        ],
        axis=3
    )

    x = tf.cast((indices % W), dtype=tf.int64)
    y = tf.cast((indices / W), dtype=tf.int64)
    loc_k = tf.stack((x, y), axis=3)

    tag_k = np.squeeze(tensor_to_numpy(tag_k))
    loc_k = np.squeeze(tensor_to_numpy(loc_k))
    val_k = np.squeeze(tensor_to_numpy(val_k))

    ans = {'tag_k': tag_k, 'loc_k': loc_k, 'val_k': val_k}
    return ans


