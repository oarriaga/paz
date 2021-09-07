import numpy as np
import tensorflow as tf
from munkres import Munkres


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
            # for tag, joint in zip(tags, joints):
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
                    # check by passing the tags and joints directly and not by
                    # removing the for loop from the function
                    update_dictionary(tags, joints, idx, default,
                                      joint_dict, tag_dict)

    return np.array([[joint_dict[i] for i in joint_dict]]).astype(np.float32)


# boxes - heatmap
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


def top_k_keypoints(boxes, tag):
    box = non_maximum_supressions(boxes)
    box = tf.transpose(box, [0, 3, 1, 2])
    num_images, num_joints = box.get_shape()[:2]
    H, W = box.get_shape()[2:4]
    box = tf.reshape(box, [num_images, num_joints, -1])

    # val-heatmaps
    # tag- tags
    # loc - location of keypoints
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

    keypoints = {'tag_k': tag_k, 'loc_k': loc_k, 'val_k': val_k}
    # loc - heatmap_location, val- heatmap_value
    return keypoints


# keypoint - person_detections
def adjust_heatmaps(boxes, keypoints):
    for batch_id, people in enumerate(keypoints):
        for people_id, i in enumerate(people):
            for joint_id, joint in enumerate(i):
                if joint[2] > 0:
                    y, x = joint[0:2]
                    xx, yy = int(x), int(y)
                    tmp = boxes[batch_id][joint_id]
                    # 
                    if tmp[xx, min(yy+1, tmp.shape[1]-1)] > \
                            tmp[xx, max(yy-1, 0)]:
                        y += 0.25
                    else:
                        y -= 0.25

                    if tmp[min(xx+1, tmp.shape[0]-1), yy] > \
                            tmp[max(0, xx-1), yy]:
                        x += 0.25
                    else:
                        x -= 0.25
                    keypoints[batch_id][people_id, joint_id, 0:2] = \
                        (y+0.5, x+0.5)
    return keypoints


def save_keypoint_tags(keypoints, tag):
    '''save tag value of detected keypoint'''
    tags = []
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0:
            x, y = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, y, x])
    return tags


def refine_heatmaps(boxes, keypoints, tag):
    if len(tag.shape) == 3:
        tag = tag[:, :, :, None]
    tags = save_keypoint_tags(keypoints, tag)
    mean_tags = np.mean(tags, axis=0)

    temp_keypoints = []
    for i in range(keypoints.shape[0]):
        tmp = boxes[i, :, :]
        tt = ((tag[i, :, :] - mean_tags[None, None, :]) ** 2).sum(axis=2)
        tmp2 = tmp - np.round(np.sqrt(tt))

        # find maximum position
        y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
        xx = x
        yy = y

        # detection score at maximum position
        val = tmp[y, x]
        x += 0.5
        y += 0.5

        if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > \
                tmp[yy, max(xx - 1, 0)]:
            x += 0.25
        else:
            x -= 0.25

        if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > \
                tmp[max(0, yy - 1), xx]:
            y += 0.25
        else:
            y -= 0.25

        temp_keypoints.append((x, y, val))
    temp_keypoints = np.array(temp_keypoints)

    if temp_keypoints is not None:
        for i in range(boxes.shape[0]):
            if temp_keypoints[i, 2] > 0 and keypoints[i, 2] == 0:
                keypoints[i, :2] = temp_keypoints[i, :2]
                keypoints[i, 2] = temp_keypoints[i, 2]
    return keypoints


def convert_to_numpy(boxes, tag):
    boxes = tensor_to_numpy(boxes)
    tag = tensor_to_numpy(tag)
    if not TAG_PER_JOINT:
        tag = np.tile(tag, ((NUM_JOINTS, 1, 1, 1)))
    return boxes, tag
