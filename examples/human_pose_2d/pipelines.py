import numpy as np
from paz import processors as pr
import processors as pe


class PreprocessImage(pr.Processor):
    def __init__(self, min_scale=1, input_size=512, inverse=False):
        super(PreprocessImage, self).__init__()
        self.get_image_center = pe.GetImageCenter()
        self.get_multi_scale_size = pe.GetMultiScaleSize(min_scale, input_size)
        self.get_affine_transform = pe.GetAffineTransform(inverse)
        self.transform_image = pr.SequentialProcessor(
            [pe.WarpAffine(), pe.ImagenetPreprocessInput(), pr.ExpandDims(0)])

    def call(self, image):
        center = self.get_image_center(image)
        scale, size = self.get_multi_scale_size(image)
        transform = self.get_affine_transform(center, scale, size)
        image = self.transform_image(image, transform, size)
        return image, center, scale


class GetMultiStageOutputs(pr.Processor):
    def __init__(self, model, dataset, data_with_center, with_AE=(True, False),
                 with_AE_loss=(True, False), with_heatmap=(True, True),
                 with_heatmap_loss=(True, True), tag_per_joint=True,
                 project2image=True, ignore_centers=True):
        super(GetMultiStageOutputs, self).__init__()
        get_joint_info = pe.GetJointInfo(dataset, data_with_center)
        num_joint, self.fliped_joint_order = get_joint_info()[1:]
        self.predict = pr.SequentialProcessor(
            [pr.Predict(model), pe.ResizeOutput('2x')])
        self.get_heatmaps = pr.SequentialProcessor(
            [pe.GetHeatmapsAverage(dataset, data_with_center, with_heatmap,
             with_heatmap_loss), pe.UpdateHeatmaps()])
        self.get_tags = pe.GetTags(with_AE_loss, with_AE, data_with_center,
                                   dataset, with_heatmap_loss, tag_per_joint)

        self.postprocess = pr.SequentialProcessor()
        if data_with_center and ignore_centers:
            self.postprocess.add(pe.RemoveLastElement())
        if project2image:
            self.postprocess.add(pe.Resize('2x'))

    def call(self, image, with_flip):
        outputs = self.predict(image)
        heatmaps = self.get_heatmaps(outputs, heatmaps=[], with_flip=False)
        tags = self.get_tags(outputs, tags=[], with_flip=False)
        if with_flip:
            outputs = self.predict(np.flip(image, [2]))
            heatmaps = self.get_heatmaps(outputs, heatmaps, with_flip)
            tags = self.get_tags(outputs, tags, with_flip)
        heatmaps = self.postprocess(heatmaps)
        tags = self.postprocess(tags)
        return heatmaps, tags


class AggregateResults(pr.Processor):
    def __init__(self, project2image=True, with_flip=False):
        super(AggregateResults, self).__init__()
        self.aggregate_tags = pr.SequentialProcessor(
            [pe.ExpandTagsDimension(), pe.Concatenate(4),
             pe.Transpose([0, 3, 1, 2, 4])])
        self.aggregate_heatmaps = pr.SequentialProcessor(
            [pe.CalculateHeatmapsAverage(with_flip),
             pe.UpdateFinalHeatmaps(project2image),
             pe.Transpose([0, 3, 1, 2])])

    def call(self, heatmaps, tags):
        tags = self.aggregate_tags(tags)
        heatmaps = self.aggregate_heatmaps(heatmaps)
        return heatmaps, tags


class HeatmapsParser(pr.Processor):
    def __init__(self, max_num_people, dataset, data_with_center,
                 tag_thresh=1, detection_thresh=0.2, tag_per_joint=True,
                 ignore_too_much=False, use_detection_val=True):
        super(HeatmapsParser, self).__init__()
        get_joint_info = pe.GetJointInfo(dataset, data_with_center)
        joint_order, num_joints = get_joint_info()[:2]

        self.group_joints = pr.SequentialProcessor()
        self.group_joints.add(pe.TopKDetections(max_num_people, tag_per_joint,
                                                num_joints))
        self.group_joints.add(pe.GroupJointsByTag(max_num_people, joint_order,
                                                  tag_thresh, detection_thresh,
                                                  ignore_too_much,
                                                  use_detection_val))

        self.adjust_joints = pe.AdjustJointsLocations()
        self.get_scores = pe.GetScores()
        self.tile_array = pe.TileArray(num_joints, tag_per_joint)
        self.refine_joints = pe.RefineJointsLocations()

    def call(self, heatmaps, tags, adjust=True, refine=True):
        grouped_joints = self.group_joints(heatmaps, tags)
        if adjust:
            grouped_joints = self.adjust_joints(heatmaps, grouped_joints)[0]
        scores = self.get_scores(grouped_joints)
        heatmaps, tags = self.tile_array(heatmaps, tags)
        if refine:
            grouped_joints = self.refine_joints(heatmaps, tags, grouped_joints)
        return [grouped_joints], scores


class TransformJoints(pr.Processor):
    def __init__(self):
        super(TransformJoints, self).__init__()
        self.get_affine_transform = pe.GetAffineTransform(inverse=True)
        self.transform_point = pe.TransformPoint()

    def call(self, grouped_joints, center, scale, heatmaps):
        transformed_joints = []
        heatmaps_size = [heatmaps.shape[3], heatmaps.shape[2]]
        for joints in grouped_joints[0]:
            transform = self.get_affine_transform(center, scale, heatmaps_size)
            for joint in joints:
                joint[0:2] = self.transform_point(joint[0:2], transform)
            transformed_joints.append(joints)
        return transformed_joints


# class GetMultiStageOutputs(pr.Processor):
#     def __init__(self, model, dataset, data_with_center, with_AE=(True, False),
#                  with_AE_loss=(True, False), with_heatmap=(True, True),
#                  with_heatmap_loss=(True, True), tag_per_joint=True,
#                  project2image=True, ignore_centers=True):
#         super(GetMultiStageOutputs, self).__init__()
#         get_joint_info = pe.GetJointInfo(dataset, data_with_center)
#         num_joint, self.fliped_joint_order = get_joint_info()[1:]
#         self.predict = pr.SequentialProcessor(
#             [pr.Predict(model), pe.ResizeOutput('2x')])
#         self.get_heatmaps = pr.SequentialProcessor(
#             [pe.GetHeatmapsAverage(num_joint, with_heatmap, with_heatmap_loss),
#              pe.UpdateHeatmaps()])
#         self.get_tags = pe.GetTags(with_AE_loss, with_AE, num_joint,
#                                    with_heatmap_loss, tag_per_joint)
#         self.flip_output = pe.FlipOutput()

#         self.postprocess = pr.SequentialProcessor()
#         if data_with_center and ignore_centers:
#             self.postprocess.add(pe.RemoveLastElement())
#         if project2image:
#             self.postprocess.add(pe.Resize('2x'))

#     def call(self, image, with_flip=False):
#         outputs = self.predict(image)
#         heatmaps = self.get_heatmaps(outputs, heatmaps=[], with_flip=False)
#         tags = self.get_tags(outputs, tags=[])
#         if with_flip:
#             outputs = self.predict(np.flip(image, [2]))
#             outputs = self.flip_output(outputs, [2])
#             indices = self.fliped_joint_order
#             heatmaps = self.get_heatmaps(outputs, heatmaps, indices, with_flip)
#             tags = self.get_tags(outputs, tags, indices, with_flip)
#         heatmaps = self.postprocess(heatmaps)
#         tags = self.postprocess(tags)
        # return heatmaps, tags