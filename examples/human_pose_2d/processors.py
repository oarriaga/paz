import numpy as np
from paz import processors as pr
import cv2
import backend as B


class ImagenetPreprocessInput(pr.Processor):
    def __init__(self):
        super(ImagenetPreprocessInput, self).__init__()

    def call(self, image):
        return B.imagenet_preprocess_input(image)


class GetImageCenter(pr.Processor):
    def __init__(self, offset=0.5):
        super(GetImageCenter, self).__init__()
        self.offset = offset

    def call(self, image):
        center_W, center_H = B.calculate_image_center(image)
        center_W = int(B.add_offset(center_W, self.offset))
        center_H = int(B.add_offset(center_H, self.offset))
        return np.array([center_W, center_H])


class GetTransformationSize(pr.Processor):
    def __init__(self, input_size):
        super(GetTransformationSize, self).__init__()
        self.input_size = input_size

    def call(self, image):
        H, W = image.shape[:2]
        if W < H:
            # for portrait image
            W, H = B.get_transformation_size(self.input_size, W, H)
        else:
            # for landscape image
            H, W = B.get_transformation_size(self.input_size, H, W)
        size = (W, H)
        return size


class GetTransformationScale(pr.Processor):
    def __init__(self, scaling_factor):
        super(GetTransformationScale, self).__init__()
        self.scaling_factor = scaling_factor

    def call(self, image, size):
        H, W = image.shape[:2]
        dims1_resized, dims2_resized = size
        if W < H:
            scale_W, scale_H = B.get_transformation_scale(
                W, dims1_resized, dims2_resized, self.scaling_factor)
        else:
            scale_H, scale_W = B.get_transformation_scale(
                H, dims2_resized, dims1_resized, self.scaling_factor)
        scale = np.array([scale_W, scale_H])
        return scale


class GetAffineTransform(pr.Processor):
    def __init__(self, inverse):
        super(GetAffineTransform, self).__init__()
        self.inverse = inverse

    def call(self, center, scale, size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])
        source_points = B.get_input_image_points(scale, center)
        output_points = B.get_output_image_points(size)
        if self.inverse:
            transform = cv2.getAffineTransform(output_points, source_points)
        else:
            transform = cv2.getAffineTransform(source_points, output_points)
        return transform


class WarpAffine(pr.Processor):
    def __init__(self):
        super(WarpAffine, self).__init__()

    def call(self, image, transform, size):
        image = cv2.warpAffine(image, transform, size)
        return image


class ScaleInput(pr.Processor):
    def __init__(self, scale_factor):
        super(ScaleInput, self).__init__()
        self.scale_factor = scale_factor

    def call(self, input):
        for arg in range(len(input)):
            H, W = input[arg].shape[-2:]
            H, W = self.scale_factor*H, self.scale_factor*W
            input[arg] = B.resize_output(input[arg], (W, H))
        return input


class TransposeOutput(pr.Processor):
    def __init__(self, axes):
        super(TransposeOutput, self).__init__()
        self.axes = axes

    def call(self, outputs):
        for arg in range(len(outputs)):
            outputs[arg] = np.transpose(outputs[arg], self.axes)
        return outputs


class ScaleOutput(pr.Processor):
    def __init__(self, scale_factor):
        super(ScaleOutput, self).__init__()
        self.scale_factor = int(scale_factor)

    def call(self, outputs):
        for arg in range(len(outputs)):
            H, W = outputs[arg].shape[-2:]
            H, W = self.scale_factor*H, self.scale_factor*W
            if len(outputs) > 1 and arg != len(outputs) - 1:
                outputs[arg] = B.resize_output(outputs[arg], (W, H))
        return outputs


class GetHeatmaps(pr.Processor):
    def __init__(self, flipped_joint_order):
        super(GetHeatmaps, self).__init__()
        self.indices = flipped_joint_order
        self.num_joint = len(flipped_joint_order)

    def call(self, outputs, with_flip):
        num_heatmaps = 0
        heatmap_sum = 0
        if not with_flip:
            for output in outputs:
                heatmap_sum = B.get_heatmap_sum(
                    output, self.num_joint, heatmap_sum)
                num_heatmaps = num_heatmaps + 1

        if with_flip:
            for output in outputs:
                heatmap_sum = B.get_heatmap_sum_with_flip(
                    output, self.num_joint, self.indices, heatmap_sum)
                num_heatmaps = num_heatmaps + 1

        heatmaps = heatmap_sum / num_heatmaps
        return heatmaps


class GetTags(pr.Processor):
    def __init__(self, flipped_joint_order):
        super(GetTags, self).__init__()
        self.indices = flipped_joint_order
        self.num_joint = len(flipped_joint_order)

    def call(self, outputs, with_flip):
        output = outputs[0]
        if not with_flip:
            tags = B.get_tags(output, self.num_joint)

        if with_flip:
            tags = B.get_tags_with_flip(
                output, self.num_joint, self.indices)
        return tags


class RemoveLastElement(pr.Processor):
    def __init__(self):
        super(RemoveLastElement, self).__init__()

    def call(self, x):
        if all(isinstance(each, list) for each in x):
            return [each[:, :-1] for each in x]
        else:
            return x[:, :-1]


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


class AggregateHeatmapsAverage(pr.Processor):
    def __init__(self, project2image):
        super(AggregateHeatmapsAverage, self).__init__()
        self.project2image = project2image

    def call(self, heatmaps_average):
        final_heatmaps = heatmaps_average
        if self.project2image:
            final_heatmaps = final_heatmaps + heatmaps_average
        return final_heatmaps


class ExpandTagsDimension(pr.Processor):
    def __init__(self):
        super(ExpandTagsDimension, self).__init__()
        self.expand_dims = pr.ExpandDims(-1)

    def call(self, tags):
        updated_tags = []
        for tag in tags:
            updated_tags.append(self.expand_dims(tag))
        return updated_tags


class TopKDetections(pr.Processor):
    def __init__(self, max_num_people):
        super(TopKDetections, self).__init__()
        self.max_num_people = max_num_people

    def call(self, heatmaps, tags):
        top_k_detections = B.top_k_detections(heatmaps, tags,
                                              self.max_num_people)
        return top_k_detections


class GroupJointsByTag(pr.Processor):
    def __init__(self, max_num_people, joint_order, tag_thresh,
                 detection_thresh):
        super(GroupJointsByTag, self).__init__()
        self.max_num_people = max_num_people
        self.joint_order = joint_order
        self.tag_thresh = tag_thresh
        self.detection_thresh = detection_thresh

    def call(self, detections):
        return B.group_joints_by_tag(detections, self.max_num_people,
                                     self.joint_order, self.detection_thresh,
                                     self.tag_thresh)


class AdjustJointsLocations(pr.Processor):
    def __init__(self):
        super(AdjustJointsLocations, self).__init__()

    def call(self, heatmaps, grouped_joints):
        grouped_joints = B.adjust_joints_locations(heatmaps, grouped_joints)
        return grouped_joints


class GetScores(pr.Processor):
    def __init__(self):
        super(GetScores, self).__init__()

    def call(self, grouped_joints):
        score = B.get_score(grouped_joints)
        return score


class RefineJointsLocations(pr.Processor):
    def __init__(self):
        super(RefineJointsLocations, self).__init__()

    def call(self, heatmaps, tags, grouped_joints):
        for arg in range(len(grouped_joints)):
            grouped_joints[arg] = B.refine_joints_locations(
                heatmaps, tags, grouped_joints[arg])
        return grouped_joints


class TransformJoints(pr.Processor):
    def __init__(self):
        super(TransformJoints, self).__init__()

    def call(self, joints, transform):
        transformed_joints = B.transform_joints(joints, transform)
        return transformed_joints


class ExtractJoints(pr.Processor):
    def __init__(self):
        super(ExtractJoints, self).__init__()

    def call(self, joints):
        joints = B.extract_joints(joints)
        return joints


class DrawSkeleton(pr.Processor):
    def __init__(self, dataset):
        super(DrawSkeleton, self).__init__()
        self.dataset = dataset

    def call(self, image, joints):
        image = B.draw_skeleton(image, joints, dataset=self.dataset)
        return image
