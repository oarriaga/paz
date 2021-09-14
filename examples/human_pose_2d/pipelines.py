import numpy as np
from paz import processors as pr
import processors as pe


class GetMultiScaleSize(pr.Processor):
    def __init__(self, min_scale, input_size):
        super(GetMultiScaleSize, self).__init__()
        self.resize = pe.ResizeDimensions(min_scale)
        self.get_image_center = pe.GetImageCenter()
        self.min_input_size = pe.MinInputSize(min_scale, input_size)

    def call(self, image):
        H, W, _ = image.shape
        center = self.get_image_center(image)
        min_input_size = self.min_input_size()
        if W < H:
            W, H, scale_W, scale_H = self.resize(min_input_size, W, H)
        else:
            H, W, scale_H, scale_W = self.resize(min_input_size, H, W)

        return (W, H), center, np.array([scale_W, scale_H])


class GetAffineTransform(pr.Processor):
    def __init__(self, inv=0):
        super(GetAffineTransform, self).__init__()
        self.inv = inv
        self.construct_source_image = pe.ConstructSourceImage()
        self.construct_output_image = pe.ConstructOutputImage()
        self.get_affine_transform = pe.GetAffineTransform()

    def call(self, center, scale, output_size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        source_image = self.construct_source_image(scale, center)
        output_image = self.construct_output_image(output_size)

        if self.inv:
            transform = self.get_affine_transform(output_image, source_image)
        else:
            transform = self.get_affine_transform(source_image, output_image)

        return transform


class ResizeAlignMultiScale(pr.Processor):
    def __init__(self, min_scale=1, input_size=512):
        super(ResizeAlignMultiScale, self).__init__()
        self.get_multi_scale_size = GetMultiScaleSize(min_scale, input_size)
        self.get_affine_transform = GetAffineTransform()
        self.warp_affine = pe.WarpAffine()

    def call(self, image):
        resized_size, center, scale = self.get_multi_scale_size(image)
        transform = self.get_affine_transform(center, scale, resized_size)
        resized_image = self.warp_affine(image, transform, resized_size)
        return resized_image, center, scale


class GetHeatmaps(pr.Processor):
    def __init__(self, num_joint, with_heatmap_loss, with_heatmap):
        super(GetHeatmaps, self).__init__()
        self.num_joint = num_joint
        self.with_heatmap_loss = with_heatmap_loss
        self.with_heatmap = with_heatmap
        self.up_sampling2D = pe.UpSampling2D(size=(2, 2),
                                             interpolation='bilinear')
        self.update_heatmaps_average = pe.UpdateHeatmapsAverage()
        self.increment_by_one = pe.IncrementByOne()
        self.update_heatmaps = pe.UpdateHeatmaps()

    def call(self, outputs, heatmaps, indices=[], with_flip=False):
        num_heatmaps = 0
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = self.up_sampling2D(output)
            if with_flip:
                output = np.flip(output, [2])

            if self.with_heatmap_loss[i] and self.with_heatmap[i]:
                heatmaps_average = self.update_heatmaps_average(output,
                                                                self.num_joint,
                                                                indices,
                                                                with_flip)
                num_heatmaps = self.increment_by_one(num_heatmaps)

        heatmaps = self.update_heatmaps(heatmaps, heatmaps_average,
                                        num_heatmaps)

        return heatmaps


class GetTags(pr.Processor):
    def __init__(self, with_AE_loss, with_AE, num_joint,
                 with_heatmap_loss, tag_per_joint):
        super(GetTags, self).__init__()
        self.with_AE_loss = with_AE_loss
        self.with_AE = with_AE
        self.calculate_offset = pe.CalculateOffset(num_joint,
                                                   with_heatmap_loss)
        self.update_tags = pe.UpdateTags(tag_per_joint)
        self.up_sampling2D = pe.UpSampling2D(size=(2, 2),
                                             interpolation='bilinear')

    def call(self, outputs, tags, indices=[], with_flip=False):
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = self.up_sampling2D(output)
            if with_flip:
                output = np.flip(output, [2])
            offset = self.calculate_offset(i)

            if self.with_AE_loss[i] and self.with_AE[i]:
                tags = self.update_tags(output, tags, offset,
                                        indices, with_flip)

        return tags


class GetMultiStageOutputs(pr.Processor):
    def __init__(self, dataset, data_with_center,
                 with_flip=True, tag_per_joint=True,
                 with_AE_loss=(True, False), with_AE=(True, False),
                 with_heatmap_loss=(True, True), with_heatmap=(True, True)):
        super(GetMultiStageOutputs, self).__init__()
        get_joint_order = pe.GetJointOrder(dataset, data_with_center)
        num_joints = len(get_joint_order())
        self.flip_joint_order = pe.FlipJointOrder(dataset, data_with_center)
        self.with_flip = with_flip
        self.get_heatmaps = GetHeatmaps(num_joints, with_heatmap_loss,
                                        with_heatmap)
        self.get_tags = GetTags(with_AE_loss, with_AE, num_joints,
                                with_heatmap_loss, tag_per_joint)
        self.flip_joint_order = pe.FlipJointOrder(dataset, data_with_center)

    def call(self, model, image):
        tags = []
        heatmaps = []

        outputs = model(image)
        heatmaps = self.get_heatmaps(outputs, heatmaps)
        tags = self.get_tags(outputs, tags)

        if self.with_flip:
            indices = self.flip_joint_order()
            outputs = model(np.flip(image, [2]))

            heatmaps = self.get_heatmaps(outputs, heatmaps, indices,
                                         self.with_flip)
            tags = self.get_tags(outputs, tags, indices, self.with_flip)
        return heatmaps, tags


class ProcessHeatmapsTags(pr.Processor):
    def __init__(self, data_with_center, ignore_centers=True,
                 project2image=True):
        super(ProcessHeatmapsTags, self).__init__()
        self.data_with_center = data_with_center
        self.ignore_centers = ignore_centers
        self.project2image = project2image
        self.remove_last_element = pe.RemoveLastElement()
        self.up_sampling2D = pe.UpSampling2D(size=(2, 2),
                                             interpolation='bilinear')

    def call(self, heatmaps, tags):
        if self.data_with_center and self.ignore_centers:
            heatmaps = self.remove_last_element(heatmaps)
            tags = self.remove_last_element(tags)

        if self.project2image:
            heatmaps = self.up_sampling2D(heatmaps)
            tags = self.up_sampling2D(tags)

        return heatmaps, tags


class AggregateResults(pr.Processor):
    def __init__(self, project2image=True, flip_test=True):
        super(AggregateResults, self).__init__()
        self.project2image = project2image
        self.flip_test = flip_test
        self.expand_dims = pr.ExpandDims(-1)
        self.heatmaps_average = pe.CalculateHeatmapsAverage()
        self.transpose = pe.Transpose()
        self.up_sampling2D = pe.UpSampling2D(size=(4, 4),
                                             interpolation='bilinear')

    def call(self, heatmaps, tags, final_heatmaps=None,
             final_tags=[]):
        if final_heatmaps is not None and not self.project2image:
            tags = self.up_sampling2D(tags)
        for tag in tags:
            final_tags.append(self.expand_dims(tag))

        if self.flip_test:
            heatmaps_average = self.heatmaps_average(heatmaps)
        else:
            heatmaps_average = heatmaps[0]

        if final_heatmaps is None:
            final_heatmaps = heatmaps_average
        elif self.project2image:
            final_heatmaps += heatmaps_average
        else:
            final_heatmaps += self.up_sampling2D(heatmaps_average)

        final_heatmaps = self.transpose(final_heatmaps, [0, 3, 1, 2])
        final_tags = np.concatenate(final_tags, axis=4)
        final_tags = self.transpose(final_tags, [0, 3, 1, 2, 4])

        return final_heatmaps, final_tags


class HeatmapsParser(pr.Processor):
    def __init__(self, max_num_people, dataset, data_with_center,
                 tag_thresh=1, detection_thresh=0.2, tag_per_joint=True,
                 ignore_too_much=False, use_detection_val=True):
        super(HeatmapsParser, self).__init__()
        get_joint_order = pe.GetJointOrder(dataset, data_with_center)
        joint_order = get_joint_order()
        num_joints = len(joint_order)
        self.tag_per_joint = tag_per_joint
        self.top_k_detections = pe.TopKDetections(max_num_people,
                                                  tag_per_joint, num_joints)
        self.group_joints_by_tag = pe.GroupJointsByTag(max_num_people,
                                                       joint_order, tag_thresh,
                                                       detection_thresh,
                                                       ignore_too_much,
                                                       use_detection_val)
        self.adjust_joints_locations = pe.AdjustJointsLocations()
        self.get_scores = pe.GetScores()
        self.tile_array = pe.TileArray(num_joints)
        self.refine_joints_locations = pe.RefineJointsLocations()

    def call(self, heatmaps, tags, adjust=True, refine=True):
        top_k_detections = self.top_k_detections(heatmaps, tags)
        grouped_joints = self.group_joints_by_tag(top_k_detections)
        if adjust:
            grouped_joints = self.adjust_joints_locations(heatmaps,
                                                          grouped_joints)[0]
        scores = self.get_scores(grouped_joints)
        heatmaps, tags = heatmaps[0], tags[0]
        if not self.tag_per_joint:
            tags = self.tile_array(tags)
        if refine:
            for arg, joints in enumerate(grouped_joints):
                joints = self.refine_joints_locations(heatmaps, joints, tags)
        return [grouped_joints], scores


class TransformJoints(pr.Processor):
    def __init__(self):
        super(TransformJoints, self).__init__()
        self.get_affine_transform = GetAffineTransform(inv=1)
        self.transform_point = pe.TransformPoint()

    def call(self, grouped_joints, center, scale, heatmaps):
        transformed_joints = []
        heatmaps_size = [heatmaps.shape[3],
                         heatmaps.shape[2]]
        for joints in grouped_joints[0]:
            transform = self.get_affine_transform(center, scale, heatmaps_size)
            for joint in joints:
                joint[0:2] = self.transform_point(joint[0:2], transform)
            transformed_joints.append(joints)
        return transformed_joints


# *************************************************************************
