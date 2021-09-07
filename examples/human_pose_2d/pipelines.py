import numpy as np
from paz import processors as pr
import processors as pe
import tensorflow as tf
import cv2


class GetMultiScaleSize(pr.Processor):
    def __init__(self):
        super(GetMultiScaleSize, self).__init__()
        self.resize = pe.ResizeDimensions()
        self.get_image_center = pe.GetImageCenter()
        self.min_input_size = pe.MinInputSize()

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
    def __init__(self, inv=0):
        super(AffineTransform, self).__init__()
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
    def __init__(self):
        super(ResizeAlignMultiScale, self).__init__()
        self.get_multi_scale_size = GetMultiScaleSize()
        self.affine_transform = AffineTransform()
        self.warp_affine = pe.WarpAffine()

    def call(self, image, current_scale):
        resized_size, center, scale = self.get_multi_scale_size(image,
                                                                current_scale)
        transform = self.affine_transform(center, scale, resized_size)
        resized_image = self.warp_affine(image, transform, resized_size)
        return resized_size, resized_image, center, scale


class CalculateHeatmapParameters(pr.Processor):
    def __init__(self, num_joint, with_heatmap_loss, test_with_heatmap,
                 with_AE_loss, test_with_AE, tag_per_joint):
        super(CalculateHeatmapParameters, self).__init__()
        self.num_joint = num_joint
        self.with_heatmap_loss = with_heatmap_loss
        self.test_with_heatmap = test_with_heatmap
        self.with_AE_loss = with_AE_loss
        self.test_with_AE = test_with_AE
        self.up_sampling2D = pe.UpSampling2D(size=(2, 2),
                                             interpolation='bilinear')
        self.calculate_offset = pe.CalculateOffset(num_joint,
                                                   with_heatmap_loss)
        self.update_tags = pe.UpdateTags(tag_per_joint)
        self.update_heatmap_average = pe.UpdateHeatmapAverage()
        self.increment_by_one = pe.IncrementByOne()

    def call(self, outputs, tags, indices=[], with_flip=False):
        num_heatmaps = 0
        heatmaps_average = 0
        for i, output in enumerate(outputs):
            if len(outputs) > 1 and i != len(outputs) - 1:
                output = self.up_sampling2D(output)

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


class GetMultiStageOutputs(pr.Processor):
    def __init__(self, with_heatmap_loss, test_with_heatmap, with_AE_loss,
                 test_with_AE, dataset, dataset_with_centers, tag_per_joint,
                 test_ignore_centers, test_scale_factor, test_project2image,
                 test_flip_test, num_joint, with_flip=False,
                 project2image=False):
        super(GetMultiStageOutputs, self).__init__()
        self.dataset = dataset
        self.with_flip = with_flip
        self.dataset_with_centers = dataset_with_centers
        self.test_ignore_centers = test_ignore_centers
        self.project2image = project2image
        self.heatmap_parameters = CalculateHeatmapParameters(num_joint,
                                        with_heatmap_loss, test_with_heatmap,
                                        with_AE_loss, test_with_AE,
                                        tag_per_joint)
        self.flip_joint_order = pe.FlipJointOrder(dataset_with_centers)
        self.update_heatmaps = pe.UpdateHeatmaps()
        self.remove_last_element = pe.RemoveLastElement()
        self.up_sampling2D = pe.UpSampling2D(size=(2, 2),
                                             interpolation='bilinear')

    def call(self, model, image, size_projected=None):
        tags = []
        heatmaps = []
        outputs = model(image)

        tags, num_heatmaps, heatmap_average = self.heatmap_parameters(outputs,
                                                                      tags)

        heatmaps = self.update_heatmaps(heatmaps, heatmap_average,
                                        num_heatmaps)

        # put this into a different class/function
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
            heatmaps = self.up_sampling2D(heatmaps)
            tags = self.up_sampling2D(tags)
        return heatmaps, tags


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
                tags = self.up_sampling2D(tags)
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


class HeatmapsParser(pr.Processor):
    def __init__(self):
        super(HeatmapsParser, self).__init__()
        self.match_by_tag = pe.MatchByTag()
        self.top_k_keypoints = pe.TopK_Keypoints()
        self.adjust_keypoints = pe.AdjustKeypoints()
        self.refine_keypoints = pe.RefineKeypoints()
        self.get_scores = pe.GetScores()
        self.convert_to_numpy = pe.ConvertToNumpy()

    def call(self, boxes, tag, adjust=True, refine=True):
        keypoints = self.top_k_keypoints(boxes, tag)
        keypoints = list(self.match_by_tag(keypoints))
        if adjust:
            keypoints = self.adjust_keypoints(boxes, keypoints)[0]
        scores = self.get_scores(keypoints)
        boxes, tag = self.convert_to_numpy(boxes[0], tag[0])
        if refine:
            for i in range(len(keypoints)):
                keypoints[i] = self.refine_keypoints(boxes, keypoints[i], tag)
        return [keypoints], scores


class FinalPrediction(pr.Processor):
    def __init__(self):
        super(FinalPrediction, self).__init__()
        self.affine_transform = AffineTransform(inv=1)
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


# *************************************************************************


coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}
coco_part_orders = [
    ('nose', 'eye_l'), ('eye_l', 'eye_r'), ('eye_r', 'nose'),
    ('eye_l', 'ear_l'), ('eye_r', 'ear_r'), ('ear_l', 'sho_l'),
    ('ear_r', 'sho_r'), ('sho_l', 'sho_r'), ('sho_l', 'hip_l'),
    ('sho_r', 'hip_r'), ('hip_l', 'hip_r'), ('sho_l', 'elb_l'),
    ('elb_l', 'wri_l'), ('sho_r', 'elb_r'), ('elb_r', 'wri_r'),
    ('hip_l', 'kne_l'), ('kne_l', 'ank_l'), ('hip_r', 'kne_r'),
    ('kne_r', 'ank_r')
]


VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
    }}


def add_joints(image, joints, color, dataset='COCO'):
    part_idx = VIS_CONFIG[dataset]['part_idx']
    part_orders = VIS_CONFIG[dataset]['part_orders']

    def link(a, b, color):
        if part_idx[a] < joints.shape[0] and part_idx[b] < joints.shape[0]:
            jointa = joints[part_idx[a]]
            jointb = joints[part_idx[b]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                    color,
                    2
                )

    # add joints
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in part_orders:
        link(pair[0], pair[1], color)

    return image


def save_valid_image(image, joints, file_name, dataset='COCO'):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for person in joints:
        color = np.random.randint(0, 255, size=3)
        color = [int(i) for i in color]
        add_joints(image, person, color, dataset=dataset)

    cv2.imwrite(file_name, image)


