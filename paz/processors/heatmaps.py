import numpy as np
import tensorflow as tf

from ..abstract import Processor
from paz import processors as pr

from ..backend.keypoints import transform_keypoint
from ..backend.image import resize_image
from ..backend.keypoints import add_offset_to_point
from ..backend.heatmaps import get_keypoints_locations, get_keypoints_heatmap
from ..backend.heatmaps import get_top_k_keypoints_numpy
from ..backend.heatmaps import get_tags_heatmap, get_valid_detections
from ..backend.standard import calculate_norm, pad_matrix, tensor_to_numpy
from ..backend.standard import compare_vertical_neighbours, gather_nd
from ..backend.standard import compare_horizontal_neighbours
from ..backend.standard import max_pooling_2d


class TransposeOutput(Processor):
    """Transpose the output of the HigherHRNet model
    # Arguments
        axes: List or tuple
        Output: List of numpy array

    """
    def __init__(self, axes):
        super(TransposeOutput, self).__init__()
        self.axes = axes

    def call(self, outputs):
        for arg in range(len(outputs)):
            outputs[arg] = np.transpose(outputs[arg], self.axes)
        return outputs


class ScaleOutput(Processor):
    """Scale the output of the HigherHRNet model
    # Arguments
        scaling_factor: Int.
        full_scaling: Boolean. If all the array of array are to be scaled.
        Output: List of numpy array

    """
    def __init__(self, scale_factor, full_scaling=False):
        super(ScaleOutput, self).__init__()
        self.scale_factor = int(scale_factor)
        self.full_scaling = full_scaling

    def _resize_output(self, output, size):
        resized_output = []
        for heatmap_arg, heatmap in enumerate(output):
            resized_heatmaps = []
            for keypoint_arg in range(len(heatmap)):
                resized = resize_image(output[heatmap_arg][keypoint_arg], size)
                resized_heatmaps.append(resized)
            resized_heatmaps = np.stack(resized_heatmaps, axis=0)
        resized_output.append(resized_heatmaps)
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
    """Get Heatmaps from the model output.
    # Arguments
        flipped_keypoint_order: List of length 17 (number of keypoints).
            Flipped list of keypoint order.
        outputs: List of numpy arrays. Output of HigherHRNet model
        with_flip: Boolean. indicates whether to flip the output

    # Returns
        heatmaps: Numpy array of shape (1, num_keypoints, H, W)
    """
    def __init__(self, flipped_keypoint_order):
        super(GetHeatmaps, self).__init__()
        self.indices = flipped_keypoint_order
        self.num_keypoints = len(flipped_keypoint_order)

    def call(self, outputs, with_flip):
        num_heatmaps = 0
        heatmap_sum = 0
        if with_flip:
            for output in outputs:
                output = np.flip(output, [3])
                heatmap_sum = heatmap_sum + get_keypoints_heatmap(
                    output, self.num_keypoints, indices=self.indices)
                num_heatmaps = num_heatmaps + 1

        if not with_flip:
            for output in outputs:
                heatmap_sum = heatmap_sum + get_keypoints_heatmap(
                    output, self.num_keypoints)
                num_heatmaps = num_heatmaps + 1

        heatmaps = heatmap_sum / num_heatmaps
        return heatmaps


class GetTags(Processor):
    """Get Tags from the model output.
    # Arguments
        flipped_keypoint_order: List of length 17 (number of keypoints).
            Flipped list of keypoint order.
        outputs: List of numpy arrays. Output of HigherHRNet model
        with_flip: Boolean. indicates whether to flip the output

    # Returns
        Tags: Numpy array of shape (1, num_keypoints, H, W)
    """
    def __init__(self, flipped_keypoint_order):
        super(GetTags, self).__init__()
        self.indices = flipped_keypoint_order
        self.num_keypoints = len(flipped_keypoint_order)

    def call(self, outputs, with_flip):
        output = outputs[0]
        if not with_flip:
            tags = get_tags_heatmap(output, self.num_keypoints)

        if with_flip:
            output = np.flip(output, [3])
            tags = get_tags_heatmap(output, self.num_keypoints, self.indices)
        return tags


class RemoveLastElement(Processor):
    """Remove last element of array
    # Arguments
        x: array or list of arrays

    """
    def __init__(self):
        super(RemoveLastElement, self).__init__()

    def call(self, x):
        if all(isinstance(each, list) for each in x):
            return [each[:, :-1] for each in x]
        else:
            return x[:, :-1]


class AggregateResults(Processor):
    """Aggregate heatmaps and tags to get final heatmaps and tags for
       processing.
    # Arguments
        heatmaps: Numpy array of shape (1, num_keypoints, H, W)
        Tags: Numpy array of shape (1, num_keypoints, H, W)

    # Returns
        heatmaps: Numpy array of shape (1, num_keypoints, H, W)
        Tags: Numpy array of shape (1, num_keypoints, H, W, 2)
    """

    def __init__(self, with_flip=False):
        super(AggregateResults, self).__init__()
        self.with_flip = with_flip

    def _expand_tags_dimension(self, tags):
        updated_tags = []
        for tag in tags:
            updated_tags.append(np.expand_dims(tag, -1))
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
        tags = np.concatenate(tags, 4)
        return heatmaps, tags


class TopKDetections(Processor):
    """Extract out the top k detections
    # Arguments
        k: Int. Maximum number of instances to be detected.
        use_numpy: Boolean. Whether to use numpy functions or tf functions.
        heatmaps: Numpy array of shape (1, num_joints, H, W)
        Tags: Numpy array of shape (1, num_joints, H, W, 2)

    # Returns
        top_k_detections: Numpy array. Contains the top k keypoints locations
                          of the detection with their value and tags.
    """
    def __init__(self, k, use_numpy=False):
        super(TopKDetections, self).__init__()
        self.k = k
        self.use_numpy = use_numpy

    def _max_pooing_2d(self, heatmaps, pool_size, strides, padding,
                       use_numpy=False):
        if use_numpy:
            heatmaps = np.squeeze(heatmaps)
            heatmaps = np.transpose(heatmaps, [2, 0, 1])
            max_heatmaps = np.zeros_like(heatmaps)
            for arg, heatmap in enumerate(heatmaps):
                max_heatmaps[arg] = max_pooling_2d(heatmap, pool_size,
                                                   strides, padding)
            max_heatmaps = np.transpose(max_heatmaps, [1, 2, 0])
            max_pooled_values = np.expand_dims(max_heatmaps, 0)
        else:
            max_pooled_values = tf.keras.layers.MaxPooling2D(
                pool_size, strides, padding)(heatmaps)
        return max_pooled_values

    def _filter_heatmaps(self, heatmaps):
        heatmaps = np.transpose(heatmaps, [0, 2, 3, 1])
        maximum_values = self._max_pooing_2d(heatmaps, pool_size=3, strides=1,
                                             padding='same',
                                             use_numpy=self.use_numpy)
        maximum_values = np.equal(maximum_values, heatmaps)
        maximum_values = maximum_values.astype(np.float32)
        filtered_heatmaps = heatmaps * maximum_values
        filtered_heatmaps = np.transpose(filtered_heatmaps, [0, 3, 1, 2])
        return filtered_heatmaps

    def _get_top_k_keypoints(self, heatmaps, k, use_numpy):
        if use_numpy:
            top_k_keypoints, indices = get_top_k_keypoints_numpy(heatmaps, k)
        else:
            top_k_keypoints, indices = tf.math.top_k(heatmaps, k)
            top_k_keypoints = np.squeeze(top_k_keypoints)
            indices = tensor_to_numpy(indices)
        return top_k_keypoints, indices

    def _get_top_k_tags(self, tags, indices):
        indices = np.expand_dims(indices, -1)
        gathered = gather_nd(tags, indices, axis=2)
        return np.squeeze(gathered)

    def call(self, heatmaps, tags):
        tags = tags.astype(np.int64)
        heatmaps = self._filter_heatmaps(heatmaps)
        num_images, keypoints_count, H, W = heatmaps.shape[:4]
        heatmaps = np.reshape(heatmaps, [num_images, keypoints_count, -1])
        tags = np.reshape(tags, [num_images, keypoints_count, W*H, -1])

        top_k_keypoints, indices = self._get_top_k_keypoints(
            heatmaps, self.k, self.use_numpy)
        top_k_tags = self._get_top_k_tags(tags, indices)
        top_k_locations = get_keypoints_locations(indices, W)

        top_k_keypoints = np.expand_dims(top_k_keypoints, axis=-1)
        top_k_detections = np.concatenate((top_k_locations,
                                           top_k_keypoints,
                                           top_k_tags), 2)
        return top_k_detections


class GroupKeypointsByTag(Processor):
    """Group the keypoints with their respective tags value.
    # Arguments
        keypoint_order: List of length 17 (number of keypoints).
        tag_thresh: Float.
        detection_thresh: Float.
        Detection: Numpy array containing the location, value and tags
                   of top k keypoints

    # Returns
        grouped_keypoints: Numpy array. keypoints grouped by tag
    """
    def __init__(self, keypoint_order, tag_thresh, detection_thresh):
        super(GroupKeypointsByTag, self).__init__()
        self.keypoint_order = keypoint_order
        self.tag_thresh = tag_thresh
        self.detection_thresh = detection_thresh
        self.munkres = pr.Munkres()

    def _update_dictionary(self, tags, keypoints, arg,
                           default, keypoint_dict, tag_dict):
        for tag, keypoint in zip(tags, keypoints):
            key = tag[0]
            keypoint_dict.setdefault(key, np.copy(default))[arg] = keypoint
            tag_dict[key] = [tag]

    def _group_tags(self, grouped_keys, tag_dict):
        grouped_tags = []
        for arg in grouped_keys:
            grouped_tags.append(np.mean(tag_dict[arg], axis=0))
        return grouped_tags

    def call(self, detections):
        keypoint_dict, tag_dict = {}, {}
        default = np.zeros((detections.shape[0], detections.shape[-1]))

        for arg, keypoint_arg in enumerate(self.keypoint_order):
            keypoints = get_valid_detections(detections[keypoint_arg],
                                             self.detection_thresh)
            tags = keypoints[:, -2:]
            if arg == 0 or len(keypoint_dict) == 0:
                self._update_dictionary(tags, keypoints, keypoint_arg,
                                        default, keypoint_dict, tag_dict)
            else:
                grouped_keys = list(keypoint_dict.keys())
                grouped_tags = self._group_tags(grouped_keys, tag_dict)
                difference = np.expand_dims(tags, 1) - np.expand_dims(
                    grouped_tags, 0)
                norm = calculate_norm(difference, order=2, axis=2)
                norm = pad_matrix(norm, padding='square', value=1e10)
                lowest_cost = self.munkres.compute(norm)
                lowest_cost = np.array(lowest_cost).astype(np.int32)

                for row_arg, col_arg in lowest_cost:
                    if norm[row_arg][col_arg] < self.tag_thresh:
                        key = grouped_keys[col_arg]
                        keypoint_dict[key][keypoint_arg] = keypoints[row_arg]
                        tag_dict[key].append(tags[row_arg])
                    else:
                        self._update_dictionary(tags, keypoints, keypoint_arg,
                                                default, keypoint_dict,
                                                tag_dict)
        grouped_keypoints = list(keypoint_dict.values())
        return [np.array(grouped_keypoints)]


class AdjustKeypointsLocations(Processor):
    """Adjust the keypoint locations by removing the margins.
    # Arguments
        heatmaps: Numpy array.
        grouped_keypoints: numpy array. keypoints grouped by tag
    """
    def __init__(self):
        super(AdjustKeypointsLocations, self).__init__()

    def call(self, heatmaps, grouped_keypoints):
        for batch_id, objects in enumerate(grouped_keypoints):
            for object_id, object in enumerate(objects):
                for keypoint_id, keypoint in enumerate(object):
                    heatmap = heatmaps[batch_id][keypoint_id]
                    if keypoint[2] > 0:
                        y, x = keypoint[0:2]
                        y = compare_vertical_neighbours(x, y, heatmap)
                        x = compare_horizontal_neighbours(x, y, heatmap)
                        grouped_keypoints[batch_id][
                            object_id, keypoint_id, 0:2] = add_offset_to_point(
                                                            (y, x), offset=0.5)
        return grouped_keypoints


class GetScores(Processor):
    """Calculate the score of the detection results.
    # Arguments
        grouped_keypoints: numpy array. keypoints grouped by tag
    """
    def __init__(self):
        super(GetScores, self).__init__()

    def call(self, grouped_keypoints):
        score = []
        for keypoint in grouped_keypoints:
            score.append(keypoint[:, 2].mean())
        return score


class RefineKeypointsLocations(Processor):
    """Refine the keypoint locations by removing the margins.
    # Arguments
        heatmaps: Numpy array.
        Tgas: Numpy array.
        grouped_keypoints: numpy array. keypoints grouped by tag
    """
    def __init__(self):
        super(RefineKeypointsLocations, self).__init__()

    def _calculate_tags_mean(self, keypoints, tags):
        keypoints_tags = []
        for arg in range(keypoints.shape[0]):
            if keypoints[arg, 2] > 0:
                x, y = keypoints[arg][:2].astype(np.int32)
                keypoints_tags.append(tags[arg, y, x])
        tags_mean = np.mean(keypoints_tags, axis=0)
        tags_mean = np.expand_dims(tags_mean, axis=[0, 1])
        return tags_mean

    def _normalize_heatmap(self, arg, tags, tags_mean, heatmap):
        normalized_tags = (tags[arg, :, :] - tags_mean)
        normalized_tags_squared_sum = (normalized_tags ** 2).sum(axis=2)
        return heatmap - np.round(np.sqrt(normalized_tags_squared_sum))

    def _find_max_position(self, heatmap_per_keypoint,
                           normalized_heatmap_per_keypoint):
        max_indices = np.argmax(normalized_heatmap_per_keypoint)
        shape = heatmap_per_keypoint.shape
        x, y = np.unravel_index(max_indices, shape)
        return x, y

    def _update_keypoints(self, keypoints, updated_keypoints, heatmaps):
        updated_keypoints = np.array(updated_keypoints)
        for i in range(heatmaps.shape[0]):
            if updated_keypoints[i, 2] > 0 and keypoints[i, 2] == 0:
                keypoints[i, :3] = updated_keypoints[i, :3]
        return keypoints

    def call(self, heatmaps, tags, grouped_keypoints):
        if len(tags.shape) == 3:
            tags = np.expand_dims(tags, -1)
        for arg in range(len(grouped_keypoints)):
            tags_mean = self._calculate_tags_mean(grouped_keypoints[arg], tags)
            updated_keypoints = []
            for keypoint_arg in range(grouped_keypoints[arg].shape[0]):
                heatmap_per_keypoint = heatmaps[keypoint_arg, :, :]
                normalized_heatmap_per_keypoint = self._normalize_heatmap(
                    keypoint_arg, tags, tags_mean, heatmap_per_keypoint)

                x, y = self._find_max_position(
                    heatmap_per_keypoint, normalized_heatmap_per_keypoint)
                max_heatmaps_value = heatmap_per_keypoint[x, y]
                x, y = add_offset_to_point((x, y), offset=0.5)
                y = compare_vertical_neighbours(x, y, heatmap_per_keypoint)
                x = compare_horizontal_neighbours(x, y, heatmap_per_keypoint)
                updated_keypoints.append((y, x, max_heatmaps_value))

            grouped_keypoints[arg] = self._update_keypoints(
                grouped_keypoints[arg], updated_keypoints, heatmaps)
        return grouped_keypoints


class TransformKeypoints(Processor):
    """Transform keypoint.

    # Arguments
        grouped_keypoints: numpy array. keypoints grouped by tag
        transform: Numpy array. Transformation matrix
    """
    def __init__(self):
        super(TransformKeypoints, self).__init__()

    def call(self, grouped_keypointss, transform):
        transformed_keypointss = []
        for keypointss in grouped_keypointss:
            for keypoints in keypointss:
                keypoints[0:2] = transform_keypoint(keypoints[0:2],
                                                    transform)[:2]
            transformed_keypointss.append(keypointss[:, :3])
        return transformed_keypointss


class ExtractKeypointsLocations(Processor):
    """Extract keypoint location.

    # Arguments
        keypoints: numpy array
    """
    def __init__(self):
        super(ExtractKeypointsLocations, self).__init__()

    def call(self, keypoints):
        for keypoints_arg in range(len(keypoints)):
            keypoints[keypoints_arg] = keypoints[keypoints_arg][:, :2]
        return keypoints
