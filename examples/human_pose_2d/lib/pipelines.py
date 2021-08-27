import numpy as np
from paz import processors as pr
import processors as pe
import tensorflow as tf


class HeatmapsParser(pr.Processor):
    def __init__(self, num_joints, joint_order, detection_thresh,
                 max_num_people, ignore_too_much, use_detection_val,
                 tag_thresh, tag_per_joint):
        super(HeatmapsParser, self).__init__()
        self.match_by_tag = pe.MatchByTag(num_joints, joint_order,
                                          detection_thresh, max_num_people,
                                          ignore_too_much, use_detection_val,
                                          tag_thresh)
        self.top_k = pe.Top_K(max_num_people, num_joints, tag_per_joint)
        self.adjust = pe.Adjust()
        self.refine = pe.Refine()
        self.get_scores = pe.GetScores()
        self.tensor_to_numpy = pe.TensorToNumpy()
        self.tile_array = pe.TileArray((num_joints, 1, 1, 1))
        self.tag_per_joint = tag_per_joint

    def call(self, det, tag, adjust=True, refine=True):
        # ans - answer
        ans = self.top_k(det, tag)
        ans = list(self.match_by_tag(ans))
        if adjust:
            ans = self.adjust(ans, det)[0]
        scores = self.get_scores(ans)
        det_numpy = self.tensor_to_numpy(det[0])
        tag_numpy = self.tensor_to_numpy(tag[0])
        if not self.tag_per_joint:
            tag_numpy = self.tile_array(tag_numpy)
        if refine:
            for i in range(len(ans)):
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
        return [ans], scores


# 
class GetMultiScaleSize(pr.Processor):
    def __init__(self, input_size, min_scale):
        super(GetMultiScaleSize, self).__init__()
        self.resize = pe.ResizeDimensions(min_scale)
        self.get_image_center = pe.GetImageCenter()
        self.min_input_size = pe.MinInputSize(input_size, min_scale)

    def call(self, image, current_scale):
        H, W, _ = image.shape
        center = self.get_image_center(image)
        min_input_size = self.min_input_size()
        if W < H:
            W, H, scale_W, scale_H = self.resize(current_scale,
                                                 min_input_size, W, H)
        else:
            H, W, scale_H, scale_W = self.resize(current_scale,
                                                 min_input_size, H, W)

        return (W, H), center, np.array([scale_W, scale_H])


class AffineTransform(pr.Processor):
    def __init__(self, rotation=0, shift=np.array([0., 0.]), inv=0):
        super(AffineTransform, self).__init__()
        self.inv = inv
        rotation_angle = np.pi * rotation / 180
        self.get_direction = pe.GetDirection(rotation_angle)
        self.updateSRC = pe.UpdateSRCMatrix(shift)
        self.updateDST = pe.UpdateDSTMatrix()
        self.get_affine_transform = pe.GetAffineTransform()

    def call(self, center, scale, output_size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale = scale * 200.0
        src_W = scale[0]
        src_dir = self.get_direction([0, src_W * -0.5])
        src = self.updateSRC(scale, center, src_dir)
        dst = self.updateDST(output_size)

        if self.inv:
            transform = self.get_affine_transform(dst, src)
        else:
            transform = self.get_affine_transform(src, dst)

        return transform


class ResizeAlignMultiScale(pr.Processor):
    def __init__(self, input_size, min_scale):
        super(ResizeAlignMultiScale, self).__init__()
        self.get_multi_scale_size = GetMultiScaleSize(input_size, min_scale)
        self.affine_transform = AffineTransform(rotation=0)
        self.warp_affine = pe.WarpAffine()

    def call(self, image, current_scale):
        resized_size, center, scale = self.get_multi_scale_size(image,
                                                                current_scale)
        transform = self.affine_transform(center, scale, resized_size)
        resized_image = self.warp_affine(image, transform, resized_size)
        return resized_image, center, scale


class GetMultiStageOutputs(pr.Processor):
    def __init__(self, with_heatmap_loss, test_with_heatmap, with_AE_loss,
                 test_with_AE, dataset, dataset_with_centers, tag_per_joint,
                 test_ignore_centers, test_scale_factor, test_project2image,
                 test_flip_test, num_joint, with_flip=False,
                 project2image=False):
        super(GetMultiStageOutputs, self).__init__()
        self.dataset = dataset
        self.with_flip = with_flip
        self.update_heatmaps = pe.UpdateHeatmaps()
        self.dataset_with_centers = dataset_with_centers
        self.test_ignore_centers = test_ignore_centers
        self.project2image = project2image
        self.heatmap_parameters = CalculateHeatmapParameters(num_joint,
                                        with_heatmap_loss, test_with_heatmap,
                                        with_AE_loss, test_with_AE,
                                        tag_per_joint)
        self.flip_joint_order = pe.FlipJointOrder(dataset_with_centers)
        self.remove_last_element = pe.RemoveLastElement()
        self.up_sampling_2D = pe.UpSampling2D(size=(2, 2),
                                              interpolation='bilinear')

    def call(self, model, image, size_projected=None):
        tags = []
        heatmaps = []
        outputs = model(image)
        tags, num_heatmaps, heatmap_average = self.heatmap_parameters(outputs,
                                                                      tags)
        heatmaps = self.update_heatmaps(heatmaps, heatmap_average,
                                        num_heatmaps)

        if self.with_flip:
            flip_index = self.flip_joint_order()
            outputs_flip = model(tf.reverse(image, [2]))

            tags, num_heatmaps, heatmap_average = \
                        self.heatmap_parameters(outputs_flip, tags, flip_index,
                                                self.with_flip)

            heatmaps = self.update_heatmaps(heatmaps, heatmap_average,
                                            num_heatmaps)

        if self.dataset_with_centers and self.test_ignore_centers:
            heatmaps = self.remove_last_element(heatmaps)
            tags = self.remove_last_element(tags)

        if self.project2image and size_projected:
            heatmaps = self.up_sampling_2D(heatmaps)
            tags = self.up_sampling_2D(tags)
        return heatmaps, tags


class CalculateHeatmapParameters(pr.Processor):
    def __init__(self, num_joint, with_heatmap_loss, test_with_heatmap,
                 with_AE_loss, test_with_AE, tag_per_joint):
        super(CalculateHeatmapParameters, self).__init__()
        self.num_joint = num_joint
        self.with_heatmap_loss = with_heatmap_loss
        self.test_with_heatmap = test_with_heatmap
        self.with_AE_loss = with_AE_loss
        self.test_with_AE = test_with_AE
        self.calculate_offset = pe.CalculateOffset(num_joint,
                                                   with_heatmap_loss)
        self.update_heatmap_average = pe.UpdateHeatmapAverage()
        self.increment_by_one = pe.IncrementByOne()
        self.update_tags = pe.UpdateTags(tag_per_joint)
        self.up_sampling_2D = pe.UpSampling2D(size=(2, 2),
                                              interpolation='bilinear')

    def call(self, outputs, tags, indices=[], with_flip=False):
        num_heatmaps = 0
        heatmaps_average = 0
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = self.up_sampling_2D(output)
            if with_flip:
                output = tf.reverse(output, [2])

            offset = self.calculate_offset(i)

            if self.with_AE_loss[i] and self.test_with_AE[i]:
                tags = self.update_tags(tags, output, offset,
                                        indices, with_flip)

            if self.with_heatmap_loss[i] and self.test_with_heatmap[i]:
                heatmap_average = self.update_heatmap_average(heatmaps_average,
                                                              output, indices,
                                                              self.num_joint,
                                                              with_flip)
                num_heatmaps = self.increment_by_one(num_heatmaps)

        return tags, num_heatmaps, heatmap_average


class AggregateResults(pr.Processor):
    def __init__(self, test_scale_factor, project2image, test_flip_test):
        super(AggregateResults, self).__init__()
        self.test_scale_factor = test_scale_factor
        self.project2image = project2image
        self.test_flip_test = test_flip_test
        self.expand_dims = pr.ExpandDims(-1)
        self.heatmaps_average = pe.CalculateHeatmapAverage()
        self.postprocess_heatmaps = pe.PostProcessHeatmaps(test_scale_factor)
        self.postprocess_tags = pe.PostProcessTags()
        self.up_sampling2D = pe.UpSampling2D(size=(4, 4),
                                             interpolation='bilinear')

    def call(self, scale_factor, heatmaps, tags, final_heatmaps=None,
             tags_list=[]):
        if scale_factor == 1 or self.test_scale_factor == 1:
            if final_heatmaps is not None and not self.project2image:
                tags = self.up_sampling_2D(tags)
            for tag in tags:
                tags_list.append(self.expand_dims(tag))

        if self.test_flip_test:
            heatmaps_average = self.heatmaps_average(heatmaps)
        else:
            heatmaps_average = heatmaps[0]

        if final_heatmaps is None:
            final_heatmaps = heatmaps_average
        elif self.project2image:
            final_heatmaps += heatmaps_average
        else:
            final_heatmaps += self.up_sampling2D(heatmaps_average)

        final_heatmaps = self.postprocess_heatmaps(final_heatmaps)
        final_tags = self.postprocess_tags(tags_list)

        return final_heatmaps, final_tags


class FinalPrediction(pr.Processor):
    def __init__(self):
        super(FinalPrediction, self).__init__()
        self.affine_transform = AffineTransform(rotation=0, inv=1)
        self.affine_transform_point = pe.AffineTransformPoint()

    def call(self, grouped_joints, center, scale, heatmap_size):
        final_result = []
        for joints in grouped_joints[0]:
            transform = self.affine_transform(center, scale, heatmap_size)
            for joint in range(joints.shape[0]):
                joints[joint, 0:2] = \
                    self.affine_transform_point(joints[joint, 0:2], transform)
            final_result.append(joints)
        return final_result
