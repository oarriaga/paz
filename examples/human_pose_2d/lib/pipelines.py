import numpy as np
from paz import processors as pr
import processors as pe


# num_joints = 16
# joint_order = [i-1 for i in [1, 2, 3, 4, 5, 6, 7, 12,
#                              13, 8, 9, 10, 11, 14, 15, 16, 17]]
# print(joint_order)


# tag_thresh = 1
# detection_thresh = 0.2
# max_num_people = 30
# ignore_too_much = False
# use_detection_val = True
# tag_per_joint = True


class Parser(pr.Processor):
    def __init__(self, num_joints, joint_order, detection_thresh,
                 max_num_people, ignore_too_much, use_detection_val,
                 tag_thresh, tag_per_joint):
        super(Parser, self).__init__()
        self.match_by_tag = pe.MatchByTag(num_joints, joint_order,
                                          detection_thresh, max_num_people,
                                          ignore_too_much, use_detection_val,
                                          tag_thresh)
        self.top_k = pe.Top_K(max_num_people, num_joints, tag_per_joint)
        self.adjust = pe.Adjust()
        self.refine = pe.Refine()
        self.get_scores = pe.GetScores()
        self.tensor_to_numpy = pe.TensorToNumpy()
        self.tiled_array = pe.TiledArray((num_joints, 1, 1, 1))
        self.tag_per_joint = tag_per_joint

    def call(self, det, tag, adjust=True, refine=True):
        ans = self.top_k(det, tag)
        ans = list(self.match_by_tag(ans))
        if adjust:
            ans = self.adjust(ans, det)[0]
        scores = self.get_scores(ans)
        det_numpy = self.tensor_to_numpy(det[0])
        tag_numpy = self.tensor_to_numpy(tag[0])
        if not self.tag_per_joint:
            tag_numpy = self.tiled_array(tag_numpy)
        if refine:
            for i in range(len(ans)):
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
        return [ans], scores


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
