import numpy as np
import tensorflow as tf

from ..abstract import Processor
from paz import processors as pr

from ..backend.keypoints import transform_point
from ..backend.image import compare_vertical_neighbours, resize_image
from ..backend.image import compare_horizontal_neighbours
from ..backend.keypoints import add_offset_to_point
from ..pipelines.munkres import Munkres


class TransposeOutput(Processor):
    def __init__(self, axes):
        super(TransposeOutput, self).__init__()
        self.axes = axes

    def call(self, outputs):
        for arg in range(len(outputs)):
            outputs[arg] = np.transpose(outputs[arg], self.axes)
        return outputs


class ScaleOutput(Processor):
    def __init__(self, scale_factor, full_scaling=False):
        super(ScaleOutput, self).__init__()
        self.scale_factor = int(scale_factor)
        self.full_scaling = full_scaling

    def _resize_output(self, output, size):
        resized_output = []
        for image_arg, image in enumerate(output):
            resized_images = []
            for joint_arg in range(len(image)):
                resized = resize_image(output[image_arg][joint_arg], size)
                resized_images.append(resized)
            resized_images = np.stack(resized_images, axis=0)
        resized_output.append(resized_images)
        resized_output = np.stack(resized_output, axis=0)
        return resized_output

    def call(self, outputs):
        for arg in range(len(outputs)):
            H, W = outputs[arg].shape[-2:]
            H, W = self.scale_factor*H, self.scale_factor*W
            if self.full_scaling:
                outputs[arg] = self._resize_output(outputs[arg], (W, H))
            else:
                if len(outputs) > 1 and arg != len(outputs) - 1:
                    outputs[arg] = self._resize_output(outputs[arg], (W, H))
        return outputs


class GetHeatmaps(Processor):
    def __init__(self, flipped_joint_order):
        super(GetHeatmaps, self).__init__()
        self.indices = flipped_joint_order
        self.num_joint = len(flipped_joint_order)

    def _get_heatmap_sum(self, output, num_joints, heatmap_sum):
        heatmap_sum = heatmap_sum + output[:, :num_joints, :, :]
        return heatmap_sum

    def _get_heatmap_sum_with_flip(self, output, num_joints,
                                   indices, heatmap_sum):
        output = np.flip(output, [3])
        heatmaps = output[:, :num_joints, :, :]
        heatmap_sum = heatmap_sum + np.take(heatmaps, indices, axis=1)
        return heatmap_sum

    def call(self, outputs, with_flip):
        num_heatmaps = 0
        heatmap_sum = 0
        if not with_flip:
            for output in outputs:
                heatmap_sum = self._get_heatmap_sum(
                    output, self.num_joint, heatmap_sum)
                num_heatmaps = num_heatmaps + 1

        if with_flip:
            for output in outputs:
                heatmap_sum = self._get_heatmap_sum_with_flip(
                    output, self.num_joint, self.indices, heatmap_sum)
                num_heatmaps = num_heatmaps + 1

        heatmaps = heatmap_sum / num_heatmaps
        return heatmaps


class GetTags(Processor):
    def __init__(self, flipped_joint_order):
        super(GetTags, self).__init__()
        self.indices = flipped_joint_order
        self.num_joint = len(flipped_joint_order)

    def _get_tags(self, output, num_joint):
        tags = output[:, num_joint:, :, :]
        return tags

    def _get_tags_with_flip(self, output, num_joint, indices):
        output = np.flip(output, [3])
        tags = output[:, num_joint:, :, :]
        tags = np.take(tags, indices, axis=1)
        return tags

    def call(self, outputs, with_flip):
        output = outputs[0]
        if not with_flip:
            tags = self._get_tags(output, self.num_joint)

        if with_flip:
            tags = self._get_tags_with_flip(
                output, self.num_joint, self.indices)
        return tags


class RemoveLastElement(Processor):
    def __init__(self):
        super(RemoveLastElement, self).__init__()

    def call(self, x):
        if all(isinstance(each, list) for each in x):
            return [each[:, :-1] for each in x]
        else:
            return x[:, :-1]


class AggregateResults(pr.Processor):
    """Aggregate heatmaps and tags to get final heatmaps and tags for
       processing.
    # Arguments
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W)

    # Returns
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W, 2)
    """

    def __init__(self, with_flip=False):
        super(AggregateResults, self).__init__()
        self.with_flip = with_flip
        self.expand_dims = pr.ExpandDims(-1)
        self.concatenate = pr.Concatenate(4)

    def _expand_tags_dimension(self, tags):
        updated_tags = []
        for tag in tags:
            updated_tags.append(self.expand_dims(tag))
        return updated_tags

    def _calculate_heatmaps_average(self, heatmaps):
        if self.with_flip:
            heatmaps_average = (heatmaps[0] + heatmaps[1])/2.0
        else:
            heatmaps_average = heatmaps[0]
        return heatmaps_average

    def call(self, heatmaps, tags):
        heatmaps_average = self._calculate_heatmaps_average(heatmaps)
        heatmaps = heatmaps_average + heatmaps_average
        tags = self._expand_tags_dimension(tags)
        tags = self.concatenate(tags)
        return heatmaps, tags


class TopKDetections(Processor):
    def __init__(self, max_num_people):
        super(TopKDetections, self).__init__()
        self.max_num_people = max_num_people

    def _tensor_to_numpy(self, tensor):
        return tensor.cpu().numpy()

    def _non_maximum_supressions(self, heatmaps):
        heatmaps = np.transpose(heatmaps, [0, 2, 3, 1])
        maximum_values = tf.keras.layers.MaxPooling2D(pool_size=3, strides=1,
                                                      padding='same')(heatmaps)
        maximum_values = np.equal(maximum_values, heatmaps)
        maximum_values = maximum_values.astype(np.float32)
        filtered_heatmaps = heatmaps * maximum_values
        return filtered_heatmaps

    def _torch_gather(self, tags, indices, gather_axis=2):
        tags = tags.astype(np.int64)
        indices = indices.astype(np.int64)
        all_indices = np.ndarray(indices.shape)
        all_indices.fill(True)
        all_indices = tf.where(all_indices)
        gather_locations = indices.flatten()
        gather_indices = []
        for axis in range(len(indices.shape)):
            if axis == gather_axis:
                gather_indices.append(gather_locations)
            else:
                gather_indices.append(all_indices[:, axis])

        gather_indices = np.stack(gather_indices, axis=-1)
        gathered = tf.gather_nd(tags, gather_indices)
        return np.reshape(gathered, indices.shape)

    def _get_top_k_heatmaps(self, heatmaps, max_num_people):
        top_k_heatmaps, indices = tf.math.top_k(heatmaps, max_num_people)
        return np.squeeze(top_k_heatmaps), self._tensor_to_numpy(indices)

    def _get_top_k_tags(self, tags, indices):
        top_k_tags = []
        for arg in range(tags.shape[3]):
            top_k_tags.append(self._torch_gather(tags[:, :, :, 0], indices))
        top_k_tags = np.stack(top_k_tags, axis=3)
        return np.squeeze(top_k_tags)

    def _get_top_k_locations(self, indices, image_width):
        x = (indices % image_width).astype(np.int64)
        y = (indices / image_width).astype(np.int64)
        top_k_locations = np.stack((x, y), axis=3)
        return np.squeeze(top_k_locations)

    def call(self, heatmaps, tags):
        heatmaps = self._non_maximum_supressions(heatmaps)
        heatmaps = np.transpose(heatmaps, [0, 3, 1, 2])
        num_images, joints_count, H, W = heatmaps.shape[:4]
        heatmaps = np.reshape(heatmaps, [num_images, joints_count, -1])
        tags = np.reshape(tags, [tags.shape[0], tags.shape[1], W*H, -1])

        top_k_heatmaps, indices = self._get_top_k_heatmaps(heatmaps,
                                                           self.max_num_people)
        top_k_tags = self._get_top_k_tags(tags, indices)
        top_k_locations = self._get_top_k_locations(indices, W)

        top_k_detections = {'top_k_tags': top_k_tags,
                            'top_k_locations': top_k_locations,
                            'top_k_heatmaps': top_k_heatmaps
                            }
        return top_k_detections


class GroupJointsByTag(Processor):
    def __init__(self, max_num_people, joint_order, tag_thresh,
                 detection_thresh):
        super(GroupJointsByTag, self).__init__()
        self.max_num_people = max_num_people
        self.joint_order = joint_order
        self.tag_thresh = tag_thresh
        self.detection_thresh = detection_thresh

    def _update_dictionary(self, tags, joints, arg,
                           default, joint_dict, tag_dict):
        for tag, joint in zip(tags, joints):
            key = tag[0]
            joint_dict.setdefault(key, np.copy(default))[arg] = joint
            tag_dict[key] = [tag]

    def _group_keys_and_tags(self, joint_dict, tag_dict, max_num_people):
        grouped_keys = list(joint_dict.keys())[:max_num_people]
        grouped_tags = [np.mean(tag_dict[arg], axis=0) for arg in grouped_keys]
        return grouped_keys, grouped_tags

    def _calculate_norm(self, joints, grouped_tags, order=2):
        difference = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
        norm = np.linalg.norm(difference, ord=order, axis=2)
        num_added, num_grouped = difference.shape[:2]
        return norm, num_added, num_grouped

    def _update_norm(self, norm, num_added, num_grouped):
        shape = (num_added, (num_added - num_grouped))
        updated_norm = np.concatenate((norm, np.zeros(shape)+1e10), axis=1)
        return updated_norm

    def _round_norm(self, norm, valid_joints):
        norm = np.round(norm) * 100 - valid_joints[:, 2:3]
        return norm

    def _shortest_L2_distance(self, cost):
        munkres = Munkres(cost)
        lowest_cost_pairs = munkres.compute()
        lowest_cost_pairs = np.array(lowest_cost_pairs).astype(np.int32)
        return lowest_cost_pairs

    def _get_valid_tags_and_joints(self, detections, joint_arg,
                                   detection_thresh):
        tags, locations, heatmaps_values = detections.values()
        joints = np.concatenate((locations[joint_arg],
                                heatmaps_values[joint_arg, :, None],
                                tags[joint_arg]), 1)
        mask = joints[:, 2] > detection_thresh
        tags = tags[joint_arg]
        valid_tags = tags[mask]
        valid_joints = joints[mask]
        return valid_tags, valid_joints

    def _extract_grouped_joints(self, joint_dict):
        grouped_joints = []
        for joint_arg in joint_dict:
            grouped_joints.append(joint_dict[joint_arg])
        grouped_joints = np.array(grouped_joints).astype(np.float32)
        return [grouped_joints]

    def call(self, detections):
        tags = detections['top_k_tags']
        joint_dict, tag_dict = {}, {}
        default = np.zeros((len(self.joint_order), tags.shape[2] + 3))

        for arg, joint_arg in enumerate(self.joint_order):
            tags, joints = self._get_valid_tags_and_joints(
                detections, joint_arg, self.detection_thresh)

            if joints.shape[0] == 0:
                continue
            if arg == 0 or len(joint_dict) == 0:
                self._update_dictionary(
                    tags, joints, joint_arg, default, joint_dict, tag_dict)
            else:
                grouped_keys, grouped_tags = self._group_keys_and_tags(
                    joint_dict, tag_dict, self.max_num_people)
                norm, num_added, num_grouped = self._calculate_norm(
                    joints, grouped_tags)
                norm_copy = np.copy(norm)
                norm = self._round_norm(norm, joints)
                if num_added > num_grouped:
                    norm = self._update_norm(norm, num_added, num_grouped)

                lowest_cost_pairs = self._shortest_L2_distance(norm)
                for row_arg, col_arg in lowest_cost_pairs:
                    if (row_arg < num_added and col_arg < num_grouped
                            and norm_copy[row_arg][col_arg] < self.tag_thresh):
                        key = grouped_keys[col_arg]
                        joint_dict[key][joint_arg] = joints[row_arg]
                        tag_dict[key].append(tags[row_arg])
                    else:
                        self._update_dictionary(tags, joints, joint_arg,
                                                default, joint_dict, tag_dict)
        grouped_joints = self._extract_grouped_joints(joint_dict)
        return grouped_joints


class AdjustJointsLocations(Processor):
    def __init__(self):
        super(AdjustJointsLocations, self).__init__()

    def call(self, heatmaps, grouped_joints):
        for batch_id, people in enumerate(grouped_joints):
            for person_id, person in enumerate(people):
                for joint_id, joint in enumerate(person):
                    heatmap = heatmaps[batch_id][joint_id]
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        y = compare_vertical_neighbours(x, y, heatmap)
                        x = compare_horizontal_neighbours(x, y, heatmap)
                        grouped_joints[batch_id][person_id, joint_id, 0:2] = \
                            add_offset_to_point((y, x), offset=0.5)
        return grouped_joints


class GetScores(Processor):
    def __init__(self):
        super(GetScores, self).__init__()

    def call(self, grouped_joints):
        score = []
        for joint in grouped_joints:
            score.append(joint[:, 2].mean())
        return score


class RefineJointsLocations(Processor):
    def __init__(self):
        super(RefineJointsLocations, self).__init__()

    def _calculate_tags_mean(self, joints, tags):
        if len(tags.shape) == 3:
            tags = tags[:, :, :, None]
        joints_tags = []
        for arg in range(joints.shape[0]):
            if joints[arg, 2] > 0:
                x, y = joints[arg][:2].astype(np.int32)
                joints_tags.append(tags[arg, y, x])
        tags_mean = np.mean(joints_tags, axis=0)
        return tags, tags_mean

    def _normalize_heatmap(self, arg, tags, tags_mean, heatmap):
        normalized_tags = (tags[arg, :, :] - tags_mean[None, None, :])
        normalized_tags_squared_sum = (normalized_tags ** 2).sum(axis=2)
        return heatmap - np.round(np.sqrt(normalized_tags_squared_sum))

    def _find_max_position(self, heatmap_per_joint,
                           normalized_heatmap_per_joint):
        max_indices = np.argmax(normalized_heatmap_per_joint)
        shape = heatmap_per_joint.shape
        x, y = np.unravel_index(max_indices, shape)
        return x, y

    def _update_joints(self, joints, updated_joints, heatmaps):
        updated_joints = np.array(updated_joints)
        for i in range(heatmaps.shape[0]):
            if updated_joints[i, 2] > 0 and joints[i, 2] == 0:
                joints[i, :3] = updated_joints[i, :3]
        return joints

    def call(self, heatmaps, tags, grouped_joints):
        for arg in range(len(grouped_joints)):
            tags, tags_mean = self._calculate_tags_mean(
                grouped_joints[arg], tags)
            updated_joints = []
            for joint_arg in range(grouped_joints[arg].shape[0]):
                heatmap_per_joint = heatmaps[joint_arg, :, :]
                normalized_heatmap_per_joint = self._normalize_heatmap(
                    joint_arg, tags, tags_mean, heatmap_per_joint)

                x, y = self._find_max_position(
                    heatmap_per_joint, normalized_heatmap_per_joint)
                max_heatmaps_value = heatmap_per_joint[x, y]
                x, y = add_offset_to_point((x, y), offset=0.5)
                y = compare_vertical_neighbours(x, y, heatmap_per_joint)
                x = compare_horizontal_neighbours(x, y, heatmap_per_joint)
                updated_joints.append((y, x, max_heatmaps_value))

            grouped_joints[arg] = self._update_joints(
                grouped_joints[arg], updated_joints, heatmaps)
        return grouped_joints


class TransformJoints(Processor):
    def __init__(self):
        super(TransformJoints, self).__init__()

    def call(self, grouped_joints, transform):
        transformed_joints = []
        for joints in grouped_joints:
            for joint in joints:
                joint[0:2] = transform_point(joint[0:2], transform)[:2]
            transformed_joints.append(joints[:, :3])
        return transformed_joints


class ExtractJoints(Processor):
    def __init__(self):
        super(ExtractJoints, self).__init__()

    def call(self, joints):
        for joints_arg in range(len(joints)):
            joints[joints_arg] = joints[joints_arg][:, :2]
        return joints
