import numpy as np
import tensorflow as tf
from paz import processors as pr
import cv2

from backend.preprocess import resize_dims, calculate_image_center
from backend.preprocess import construct_source_image, construct_output_image
from backend.preprocess import calculate_min_input_size, tf_preprocess_input
from backend.heatmaps import tile_array
from backend.heatmaps import refine_joints_locations, top_k_detections
from backend.heatmaps import group_joints_by_tag, adjust_joints_locations
from backend.postprocess import draw_skeleton
from dataset import FLIP_CONFIG, JOINT_CONFIG


class TfPreprocessInput(pr.Processor):
    def __init__(self):
        super(TfPreprocessInput, self).__init__()

    def call(self, image):
        return tf_preprocess_input(image)


# ********************GetMultiScaleSize*****************************


class ResizeDimensions(pr.Processor):
    def __init__(self, min_scale):
        super(ResizeDimensions, self).__init__()
        self.min_scale = min_scale

    def call(self, min_input_size, dims1, dims2):
        dims1_resized, dims2_resized, scale_dims1, scale_dims2 = \
            resize_dims(min_input_size, dims1, dims2, self.min_scale)
        return dims1_resized, dims2_resized, scale_dims1, scale_dims2


class GetImageCenter(pr.Processor):
    def __init__(self, offset=0.5):
        super(GetImageCenter, self).__init__()
        self.offset = offset

    def call(self, image):
        center_W, center_H = calculate_image_center(image)
        center_W = int(center_W + self.offset)
        center_H = int(center_H + self.offset)
        return np.array([center_W, center_H])


class MinInputSize(pr.Processor):
    def __init__(self, min_scale, input_size):
        super(MinInputSize, self).__init__()
        self.min_scale = min_scale
        self.input_size = input_size

    def call(self):
        min_input_size = calculate_min_input_size(self.min_scale,
                                                  self.input_size)
        return min_input_size


# ********************AffineTransform*****************************


class ConstructSourceImage(pr.Processor):
    def __init__(self):
        super(ConstructSourceImage, self).__init__()

    def call(self, scale, center):
        source_image = construct_source_image(scale, center)
        return source_image


class ConstructOutputImage(pr.Processor):
    def __init__(self):
        super(ConstructOutputImage, self).__init__()

    def call(self, output_size):
        output_image = construct_output_image(output_size)
        return output_image


class GetAffineTransform(pr.Processor):
    def __init__(self):
        super(GetAffineTransform, self).__init__()

    def call(self, dst, src):
        transform = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return transform


# ********************ResizeAlignMultiScale*****************************


class WarpAffine(pr.Processor):
    def __init__(self):
        super(WarpAffine, self).__init__()

    def call(self, image, transform, size_resized):
        image_resized = cv2.warpAffine(image, transform, size_resized)
        return image_resized


# ********************CalculateHeatmapParameters*****************************


class UpSampling2D(pr.Processor):
    def __init__(self, size, interpolation):
        super(UpSampling2D, self).__init__()
        self.size = size
        self.interpolation = interpolation

    def call(self, x):
        if isinstance(x, list):
            x = [tf.keras.layers.UpSampling2D(size=self.size,
                 interpolation=self.interpolation)(each) for each in x]
            # x = [np.kron(each, np.ones(self.size)) for each in x]
        else:
            x = \
             tf.keras.layers.UpSampling2D(size=self.size,
                                          interpolation=self.interpolation)(x)
            # x = np.kron(x, np.ones(self.size))
        return x


class CalculateOffset(pr.Processor):
    def __init__(self, num_joints, loss_with_heatmap_loss):
        super(CalculateOffset, self).__init__()
        self.num_joints = num_joints
        self.loss_with_heatmap_loss = loss_with_heatmap_loss

    def call(self, idx):
        if self.loss_with_heatmap_loss[idx]:
            offset = self.num_joints
        else:
            offset = 0
        return offset


class UpdateTags(pr.Processor):
    def __init__(self, tag_per_joint):
        super(UpdateTags, self).__init__()
        self.tag_per_joint = tag_per_joint

    def call(self, output, tags, offset, indices, with_flip=False):
        tags.append(output[:, :, :, offset:])
        if with_flip and self.tag_per_joint:
            tags[-1] = np.take(tags[-1], indices, axis=-1)
        return tags


class UpdateHeatmapsAverage(pr.Processor):
    def __init__(self):
        super(UpdateHeatmapsAverage, self).__init__()

    def call(self, output, num_joints, indices, with_flip=False):
        heatmaps_average = 0
        if not with_flip:
            heatmaps_average += output[:, :, :, :num_joints]
        else:
            temp = output[:, :, :, :num_joints]
            heatmaps_average += np.take(temp, indices, axis=-1)
        return heatmaps_average


class IncrementByOne(pr.Processor):
    def __init__(self):
        super(IncrementByOne, self).__init__()

    def call(self, x):
        x += 1
        return x


# ********************GetMultiStageOutputs*****************************


class UpdateHeatmaps(pr.Processor):
    def __init__(self):
        super(UpdateHeatmaps, self).__init__()

    def call(self, heatmaps, heatmaps_average, num_heatmaps):
        heatmaps.append(heatmaps_average/num_heatmaps)
        return heatmaps


class GetJointOrder(pr.Processor):
    def __init__(self, dataset, with_center):
        super(GetJointOrder, self).__init__()
        self.dataset = dataset
        self.with_center = with_center

    def call(self):
        if not self.with_center:
            joint_order = JOINT_CONFIG[self.dataset]
        else:
            joint_order = JOINT_CONFIG[self.dataset + '_WITH_CENTER']
        return joint_order


class FlipJointOrder(pr.Processor):
    def __init__(self, dataset, with_center):
        super(FlipJointOrder, self).__init__()
        self.dataset = dataset
        self.with_center = with_center

    def call(self):
        if not self.with_center:
            joint_order = FLIP_CONFIG[self.dataset]
        else:
            joint_order = FLIP_CONFIG[self.dataset + '_WITH_CENTER']
        return joint_order


class RemoveLastElement(pr.Processor):
    def __init__(self):
        super(RemoveLastElement, self).__init__()

    def call(self, nested_list):
        return [each_list[:, :-1] for each_list in nested_list]


# **************************AggregateResults*********************************


class CalculateHeatmapsAverage(pr.Processor):
    def __init__(self):
        super(CalculateHeatmapsAverage, self).__init__()

    def call(self, heatmaps):
        heatmaps_average = (heatmaps[0] + heatmaps[1])/2.0
        return heatmaps_average


class Transpose(pr.Processor):
    def __init__(self):
        super(Transpose, self).__init__()

    def call(self, tags, permutes=None):
        tags = np.transpose(tags, permutes)
        return tags

# **************************HeatmapsParser*********************************


class TopKDetections(pr.Processor):
    def __init__(self, max_num_people, tag_per_joint, num_joints):
        super(TopKDetections, self).__init__()
        self.max_num_people = max_num_people
        self.tag_per_joint = tag_per_joint
        self.num_joints = num_joints

    def call(self, heatmaps, tags):
        keypoints = top_k_detections(heatmaps, tags, self.max_num_people,
                                     self.tag_per_joint, self.num_joints)
        return keypoints


class GroupJointsByTag(pr.Processor):
    def __init__(self, max_num_people, joint_order, tag_thresh,
                 detection_thresh, ignore_too_much, use_detection_val):
        super(GroupJointsByTag, self).__init__()
        self.max_num_people = max_num_people
        self.joint_order = joint_order
        self.tag_thresh = tag_thresh
        self.detection_thresh = detection_thresh
        self.ignore_too_much = ignore_too_much
        self.use_detection_val = use_detection_val

    def call(self, detections):
        return group_joints_by_tag(detections, self.max_num_people,
                                   self.joint_order, self.tag_thresh,
                                   self.detection_thresh, self.ignore_too_much,
                                   self.use_detection_val)


class AdjustJointsLocations(pr.Processor):
    def __init__(self):
        super(AdjustJointsLocations, self).__init__()

    def call(self, heatmaps, detections):
        detections = adjust_joints_locations(heatmaps, detections)
        return detections


class GetScores(pr.Processor):
    def __init__(self):
        super(GetScores, self).__init__()

    def call(self, ans):
        score = [i[:, 2].mean() for i in ans]
        return score


class TileArray(pr.Processor):
    def __init__(self, num_joints):
        super(TileArray, self).__init__()
        self.num_joints = num_joints

    def call(self, tags):
        return tile_array(tags, self.num_joints)


class RefineJointsLocations(pr.Processor):
    def __init__(self):
        super(RefineJointsLocations, self).__init__()

    def call(self, boxes, keypoints, tag):
        keypoints = refine_joints_locations(boxes, keypoints, tag)
        return keypoints


# **************************FinalPrediction****************************


class TransformPoint(pr.Processor):
    def __init__(self):
        super(TransformPoint, self).__init__()

    def call(self, point, transform):
        point = np.array([point[0], point[1], 1.]).T
        point_transformed = np.dot(transform, point)
        return point_transformed[:2]


# **************************SaveResult*********************************


class DrawSkeleton(pr.Processor):
    def __init__(self, dataset='COCO'):
        super(DrawSkeleton, self).__init__()
        self.dataset = dataset

    def call(self, image, joints):
        image = draw_skeleton(image, joints, dataset=self.dataset)
        return image
