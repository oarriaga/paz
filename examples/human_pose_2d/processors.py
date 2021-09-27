import numpy as np
from paz import processors as pr
import cv2
import backend as B
from dataset import get_joint_info


class ImagenetPreprocessInput(pr.Processor):
    def __init__(self):
        super(ImagenetPreprocessInput, self).__init__()

    def call(self, image):
        return B.imagenet_preprocess_input(image)


# ********************PreprocessImage*****************************


class GetImageCenter(pr.Processor):
    def __init__(self, offset=0.5):
        super(GetImageCenter, self).__init__()
        self.offset = offset

    def call(self, image):
        center_W, center_H = B.calculate_image_center(image)
        center_W = int(center_W + self.offset)
        center_H = int(center_H + self.offset)
        return np.array([center_W, center_H])


class MinInputSize(pr.Processor):
    def __init__(self, min_scale, input_size):
        super(MinInputSize, self).__init__()
        self.min_scale = min_scale
        self.input_size = input_size

    def call(self):
        min_input_size = B.calculate_min_input_size(self.min_scale,
                                                    self.input_size)
        return min_input_size


class GetMultiScaleSize(pr.Processor):
    def __init__(self, min_scale, input_size):
        super(GetMultiScaleSize, self).__init__()
        self.min_scale = min_scale
        self.min_input_size = int(((min_scale * input_size) + (64-1)) // 64*64)

    def call(self, image):
        H, W = image.shape[:2]
        if W < H:
            W, H, scale_W, scale_H = B.resize_dims(self.min_input_size, W, H,
                                                   self.min_scale)
        else:
            H, W, scale_H, scale_W = B.resize_dims(self.min_input_size, H, W,
                                                   self.min_scale)
        scale = np.array([scale_W, scale_H])
        size = (W, H)
        return scale, size


class GetAffineTransform(pr.Processor):
    def __init__(self, inverse):
        super(GetAffineTransform, self).__init__()
        self.inverse = inverse

    def call(self, center, scale, size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])
        source_image = B.construct_source_image(scale, center)
        output_image = B.construct_output_image(size)
        if self.inverse:
            transform = cv2.getAffineTransform(output_image, source_image)
        else:
            transform = cv2.getAffineTransform(source_image, output_image)
        return transform


class WarpAffine(pr.Processor):
    def __init__(self):
        super(WarpAffine, self).__init__()

    def call(self, image, transform, size):
        image = cv2.warpAffine(image, transform, size)
        return image


# ********************GetMultiStageOutputs*****************************


class Resize(pr.Processor):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = int(size[:-1])

    def call(self, outputs):
        for arg in range(len(outputs)):
            H, W = outputs[arg].shape[1:3]
            H, W = self.size*H, self.size*W
            outputs[arg] = B.resize_output(outputs[arg], (W, H))
        return outputs


class ResizeOutput(pr.Processor):
    def __init__(self, size):
        super(ResizeOutput, self).__init__()
        self.size = int(size[:-1])

    def call(self, outputs):
        for arg in range(len(outputs)):
            H, W = outputs[arg].shape[1:3]
            H, W = self.size*H, self.size*W
            if len(outputs) > 1 and arg != len(outputs) - 1:
                outputs[arg] = B.resize_output(outputs[arg], (W, H))
        return outputs


class GetHeatmapsAverage(pr.Processor):
    def __init__(self, data_info, with_heatmap, with_heatmap_loss):
        super(GetHeatmapsAverage, self).__init__()
        self.with_heatmap_loss = with_heatmap_loss
        self.with_heatmap = with_heatmap
        self.num_joint, self.fliped_joint_order = get_joint_info(data_info)[1:]

    def call(self, outputs, heatmaps, with_flip, indices=[]):
        num_heatmaps = 0
        for arg, output in enumerate(outputs):
            if (with_flip and
                    self.with_heatmap_loss[arg] and self.with_heatmap[arg]):
                output = np.flip(output, [2])
                indices = self.fliped_joint_order
                heatmaps_average = B.get_heatmaps_average(output, self.num_joint,
                                                        with_flip, indices)
                num_heatmaps += 1

            if (not with_flip and
                    self.with_heatmap_loss[arg] and self.with_heatmap[arg]):
                heatmaps_average = B.get_heatmaps_average(output, self.num_joint,
                                                        with_flip, indices)
                num_heatmaps += 1
        return heatmaps, heatmaps_average, num_heatmaps


class UpdateHeatmaps(pr.Processor):
    def __init__(self):
        super(UpdateHeatmaps, self).__init__()

    def call(self, heatmaps, heatmaps_average, num_heatmaps):
        heatmaps.append(heatmaps_average/num_heatmaps)
        return heatmaps


class GetTags(pr.Processor):
    def __init__(self, with_AE_loss, with_AE, data_info,
                 with_heatmap_loss, tag_per_joint):
        super(GetTags, self).__init__()
        self.with_AE_loss = with_AE_loss
        self.with_AE = with_AE
        self.with_heatmap_loss = with_heatmap_loss
        self.tag_per_joint = tag_per_joint
        self.num_joint, self.fliped_joint_order = get_joint_info(data_info)[1:]

    def call(self, outputs, tags, with_flip, indices=[]):
        for arg, output in enumerate(outputs):
            offset = B.calculate_offset(self.with_heatmap_loss[arg],
                                        self.num_joint)
            if with_flip and self.with_AE_loss[arg] and self.with_AE[arg]:
                output = np.flip(output, [2])
                indices = self.fliped_joint_order
                tags = B.get_tags(output, tags, offset, indices,
                                  self.tag_per_joint, with_flip)
            if not with_flip and self.with_AE_loss[arg] and self.with_AE[arg]:
                tags = B.get_tags(output, tags, offset, indices,
                                  self.tag_per_joint, with_flip)

        return tags


class RemoveLastElement(pr.Processor):
    def __init__(self):
        super(RemoveLastElement, self).__init__()

    def call(self, x):
        if all(isinstance(each, list) for each in x):
            return [each[:, :-1] for each in x]
        else:
            return x[:, :-1]


# **************************AggregateResults*********************************


class CalculateHeatmapsAverage(pr.Processor):
    def __init__(self, with_flip):
        super(CalculateHeatmapsAverage, self).__init__()
        self.with_flip = with_flip

    def call(self, heatmaps):
        if self.with_flip:
            heatmaps_average = (heatmaps[0] + heatmaps[1])/2.0
        else:
            heatmaps_average = heatmaps[0]
        return heatmaps_average


class Transpose(pr.Processor):
    def __init__(self, permutes=None):
        super(Transpose, self).__init__()
        self.permutes = permutes

    def call(self, x):
        x = np.transpose(x, self.permutes)
        return x


class Concatenate(pr.Processor):
    def __init__(self, axis):
        super(Concatenate, self).__init__()
        self.axis = axis

    def call(self, x):
        x = np.concatenate(x, self.axis)
        return x


class ExpandTagsDimension(pr.Processor):
    def __init__(self):
        super(ExpandTagsDimension, self).__init__()
        self.expand_dims = pr.ExpandDims(-1)

    def call(self, tags):
        updated_tags = []
        for tag in tags:
            updated_tags.append(self.expand_dims(tag))
        return updated_tags


class AggregateHeatmapsAverage(pr.Processor):
    def __init__(self, project2image):
        super(AggregateHeatmapsAverage, self).__init__()
        self.project2image = project2image

    def call(self, heatmaps_average):
        final_heatmaps = heatmaps_average
        if self.project2image:
            final_heatmaps += heatmaps_average
        return final_heatmaps


# **************************HeatmapsParser*********************************


class TopKDetections(pr.Processor):
    def __init__(self, max_num_people, tag_per_joint, data_info):
        super(TopKDetections, self).__init__()
        self.max_num_people = max_num_people
        self.tag_per_joint = tag_per_joint
        self.num_joints = get_joint_info(data_info)[1]

    def call(self, heatmaps, tags):
        keypoints = B.top_k_detections(heatmaps, tags, self.max_num_people,
                                       self.tag_per_joint, self.num_joints)
        return keypoints


class GroupJointsByTag(pr.Processor):
    def __init__(self, max_num_people, data_info, tag_thresh,
                 detection_thresh, ignore_too_much, use_detection_val):
        super(GroupJointsByTag, self).__init__()
        self.max_num_people = max_num_people
        self.joint_order = get_joint_info(data_info)[0]
        self.tag_thresh = tag_thresh
        self.detection_thresh = detection_thresh
        self.ignore_too_much = ignore_too_much
        self.use_detection_val = use_detection_val

    def call(self, detections):
        return B.group_joints_by_tag(detections, self.max_num_people,
                                     self.joint_order, self.detection_thresh,
                                     self.tag_thresh, self.ignore_too_much,
                                     self.use_detection_val)


class AdjustJointsLocations(pr.Processor):
    def __init__(self):
        super(AdjustJointsLocations, self).__init__()

    def call(self, heatmaps, detections):
        detections = B.adjust_joints_locations(heatmaps, detections)
        return detections


class GetScores(pr.Processor):
    def __init__(self):
        super(GetScores, self).__init__()

    def call(self, ans):
        score = [i[:, 2].mean() for i in ans]
        return score


class TileArray(pr.Processor):
    def __init__(self, data_info, tag_per_joint):
        super(TileArray, self).__init__()
        self.num_joints = get_joint_info(data_info)[1]
        self.tag_per_joint = tag_per_joint

    def call(self, heatmaps, tags):
        heatmaps, tags = heatmaps[0], tags[0]
        if not self.tag_per_joint:
            tags = self.tile_array(tags)
        return heatmaps, tags


class RefineJointsLocations(pr.Processor):
    def __init__(self):
        super(RefineJointsLocations, self).__init__()

    def call(self, heatmaps, tags, grouped_joints):
        for arg in range(len(grouped_joints)):
            grouped_joints[arg] = B.refine_joints_locations(heatmaps, tags,
                                                            grouped_joints[arg])
        return grouped_joints


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
    def __init__(self, data_info):
        super(DrawSkeleton, self).__init__()
        self.dataset = data_info['data']

    def call(self, image, joints):
        image = B.draw_skeleton(image, joints, dataset=self.dataset)
        return image
