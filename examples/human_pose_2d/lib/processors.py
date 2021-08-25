import numpy as np
import tensorflow as tf
from paz import processors as pr
from munkres import Munkres
import cv2


class LoadModel(pr.Processor):
    def __init__(self):
        super(LoadModel, self).__init__()

    def call(self, model_path):
        return tf.keras.models.load_model(model_path)


class CreateDirectory(pr.Processor):
    def __init__(self):
        super(CreateDirectory, self).__init__()

    def call(self, directory):
        print('=> creating {}'.format(directory))
        directory.mkdir(parents=True, exist_ok=True)


class ReplaceText(pr.Processor):
    def __init__(self, text):
        super(ReplaceText, self).__init__()
        self.text = text

    def call(self, oldvalue, newvalue):
        return self.text.replace(oldvalue, newvalue)


class NonMaximumSuppression(pr.Processor):
    def __init__(self, permutes, pool_size, strides, padding='same'):
        super(NonMaximumSuppression, self).__init__()
        self.permutes = permutes
        self.MaxPooling2D = tf.keras.layers.MaxPooling2D(pool_size,
                                                         strides, padding)

    def call(self, det):
        det = tf.transpose(det, self.permutes)
        maxm = self.MaxPooling2D(det)
        maxm = tf.math.equal(maxm, det)
        maxm = tf.cast(maxm, tf.float32)
        det = det * maxm
        return det


class MatchByTag(pr.Processor):
    def __init__(self, num_joints, joint_order, detection_thresh,
                 max_num_people, ignore_too_much, use_detection_val,
                 tag_thresh):
        super(MatchByTag, self).__init__()
        self.num_joints = num_joints
        self.joint_order = joint_order
        self.detection_thresh = detection_thresh
        self.max_num_people = max_num_people
        self.ignore_too_much = ignore_too_much
        self.use_detection_val = use_detection_val
        self.tag_thresh = tag_thresh
        self.joint_dict = {}
        self.tag_dict = {}

    def update_dictionary(self, tags, joints, idx, default):
        for tag, joint in zip(tags, joints):
            key = tag[0]
            self.joint_dict.setdefault(key, np.copy(default))[idx] = joint
            self.tag_dict[key] = [tag]

    def group_keys_and_tags(self, joint_dict, tag_dict, idx):
        grouped_keys = list(joint_dict.keys())[:self.max_num_people]
        grouped_tags = [np.mean(tag_dict[idx], axis=0) for idx in grouped_keys]
        return grouped_keys, grouped_tags

    def calculate_norm(self, joints, grouped_tags, order=2):
        difference = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
        norm = np.linalg.norm(difference, ord=order, axis=2)
        return difference, norm

    def concatenate_zeros(self, metrix, shape):
        concatenated = np.concatenate((metrix, np.zeros(shape)+1e10), axis=1)
        return concatenated

    def shortest_L2_distance(self, scores):
        munkres = Munkres()
        lowest_cost_pairs = munkres.compute(scores)
        lowest_cost_pairs = np.array(lowest_cost_pairs).astype(np.int32)
        return lowest_cost_pairs

    def call(self, input_):
        tag_k, loc_k, val_k = input_.values()
        # print('tag_k_shape:', tag_k.shape)
        # print('loc_k_shape:', loc_k.shape)
        # print('val_k_shape:', val_k.shape)
        tag_k = tag_k[0, :, :, :]
        loc_k = loc_k[0, :, :, :]
        val_k = val_k[0, :, :]
        # print('tag_k_shape:', tag_k.shape)
        # print('loc_k_shape:', loc_k.shape)
        # print('val_k_shape:', val_k.shape)

        default = np.zeros((self.num_joints, tag_k.shape[2] + 3))
        for i in range(self.num_joints):
            idx = self.joint_order[i]
            tags = tag_k[idx]
            joints = np.concatenate((loc_k[idx], val_k[idx, :, None],
                                    tag_k[idx]), 1)
            mask = joints[:, 2] > self.detection_thresh
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if i == 0 or len(self.joint_dict) == 0:
                self.update_dictionary(tags, joints, idx, default)

            else:
                grouped_keys, grouped_tags = self.group_keys_and_tags(
                                             self.joint_dict, self.tag_dict, i)

                if self.ignore_too_much and len(grouped_keys) == \
                   self.max_num_people:
                    continue

                difference, norm = self.calculate_norm(joints, grouped_tags)
                norm_copy = np.copy(norm)

                num_added = difference.shape[0]
                num_grouped = difference.shape[1]

                if num_added > num_grouped:
                    norm = self.concatenate_zeros(norm, (num_added,
                                                  num_added - num_grouped))

                lowest_cost_pairs = self.shortest_L2_distance(norm)

                for row, col in lowest_cost_pairs:
                    if (
                        row < num_added
                        and col < num_grouped
                        and norm_copy[row][col] < self.tag_thresh
                    ):
                        key = grouped_keys[col]
                        self.joint_dict[key][idx] = joints[row]
                        self.tag_dict[key].append(tags[row])
                    else:
                        self.update_dictionary(tags[row], joints[row], idx, default)

        return np.array([[self.joint_dict[i]
                        for i in self.joint_dict]]).astype(np.float32)


class TorchGather(pr.Processor):
    def __init__(self, gather_axis):
        super(TorchGather, self).__init__()
        self.gather_axis = gather_axis

    def call(self, x, indices):
        x = tf.cast(x, tf.int64)
        indices = tf.cast(indices, tf.int64)
        all_indices = tf.where(tf.fill(indices.shape, True))
        gather_locations = tf.reshape(indices, [indices.shape.num_elements()])
        gather_indices = []
        for axis in range(len(indices.shape)):
            if axis == self.gather_axis:
                gather_indices.append(gather_locations)
            else:
                gather_indices.append(all_indices[:, axis])

        gather_indices = tf.stack(gather_indices, axis=-1)
        gathered = tf.gather_nd(x, gather_indices)
        return tf.reshape(gathered, indices.shape)


class Top_K(pr.Processor):
    def __init__(self, max_num_people, num_joints, tag_per_joint):
        super(Top_K, self).__init__()
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.tag_per_joint = tag_per_joint
        self.torch_gather = TorchGather(2)
        self.nms = NonMaximumSuppression([0, 2, 3, 1], 3, 1)
        # det = self.nms(det)
        # self.det = tf.transpose(det, [0, 3, 1, 2])
        # self.num_images = det.get_shape()[0]
        # self.num_joints = det.get_shape()[1]
        # self.h = det.get_shape()[2]
        # self.w = det.get_shape()[3]

    def call(self, det, tag):
        det = self.nms(det)
        det = tf.transpose(det, [0, 3, 1, 2])
        num_images = det.get_shape()[0]
        num_joints = det.get_shape()[1]
        H = det.get_shape()[2]
        W = det.get_shape()[3]
        det = tf.reshape(det, [num_images, num_joints, -1])
        val_k, indices = tf.math.top_k(det, self.max_num_people)
        tag = tf.reshape(tag, [tag.get_shape()[0],
                         tag.get_shape()[1], W*H, -1])

        if not self.tag_per_joint:
            tag = tag.expand(-1, self.num_joints, -1, -1)

        tag_k = tf.stack(
            [
                self.torch_gather(tag[:, :, :, 0], indices)
                for i in range(tag.get_shape()[3])
            ],
            axis=3
        )

        x = tf.cast((indices % W), dtype=tf.int64)
        y = tf.cast((indices / W), dtype=tf.int64)

        ind_k = tf.stack((x, y), axis=3)
        ans = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }
        return ans


class Adjust(pr.Processor):
    def __init__(self):
        super(Adjust, self).__init__()

    def call(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)] > \
                           tmp[xx, max(yy-1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy] > \
                           tmp[max(0, xx-1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = \
                            (y+0.5, x+0.5)
        return ans


class Refine(pr.Processor):
    def __init__(self):
        super(Refine, self).__init__()

    def call(self, det, tag, keypoints):
        if len(tag.shape) == 3:
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)      # loc of keypoint
                tags.append(tag[i, y, x])

        prev_tag = np.mean(tags, axis=0)

        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            # from item index find 2D index
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > \
               tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > \
               tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]
        return keypoints


class GetScores(pr.Processor):
    def __init__(self):
        super(GetScores, self).__init__()

    def call(self, ans):
        return [i[:, 2].mean() for i in ans]


class TensorToNumpy(pr.Processor):
    def __init__(self):
        super(TensorToNumpy, self).__init__()

    def call(self, tensor):
        return tensor.cpu().numpy()

# tile array
# reps
class TiledArray(pr.Processor):
    def __init__(self, reps):
        super(TiledArray, self).__init__()
        self.reps = reps

    def call(self, array):
        return np.tile(array, self.reps)

# put constants in init
class ResizeDimensions(pr.Processor):
    def __init__(self, min_scale):
        super(ResizeDimensions, self).__init__()
        self.min_scale = min_scale

    def call(self, current_scale, min_input_size, D1, D2):
        D1_resized = int(min_input_size * current_scale / self.min_scale)
        D2_resized = int(int((min_input_size / D1*D2 + 63) // 64*64) *
                         current_scale/self.min_scale)
        scale_D1 = D1 / 200.0
        scale_D2 = D2_resized / D1_resized * D1 / 200.0
        return D1_resized, D2_resized, scale_D1, scale_D2


class GetImageCenter(pr.Processor):
    def __init__(self):
        super(GetImageCenter, self).__init__()

    def call(self, image):
        # input.shape[:2]
        H, W, _ = image.shape
        # create variable center_W and H
        # seperation of variables
        return np.array([int((W / 2.0) + 0.5), int((H / 2.0) + 0.5)])


class MinInputSize(pr.Processor):
    def __init__(self, input_size, min_scale):
        super(MinInputSize, self).__init__()
        self.input_size = input_size
        self.min_scale = min_scale

    def call(self):
        return int((self.min_scale * self.input_size + 63)//64 * 64)


class GetDirection(pr.Processor):
    # rotation
    def __init__(self, rotation_angle):
        super(GetDirection, self).__init__()
        self.rotation_angle = rotation_angle

    def call(self, point):
    # def call(self, point2D):
        sn, cs = np.sin(self.rotation_angle), np.cos(self.rotation_angle)
        result = [0, 0]
        # x_rotated, y_rotated
        result[0] = point[0] * cs - point[1] * sn
        result[1] = point[0] * sn + point[1] * cs
        return result


class Get3rdPoint(pr.Processor):
    def __init__(self):
        super(Get3rdPoint, self).__init__()

    def call(self, p1, p2):
    # def call(self, point2D_a, point2D_b):
        difference = p1 - p2
        return p2 + np.array([-difference[1], difference[0]], dtype=np.float32)


class UpdateSRCMatrix(pr.Processor):
    def __init__(self, shift):
        super(UpdateSRCMatrix, self).__init__()
        self.shift = shift
        self.get_3rd_point = Get3rdPoint()

    def call(self, scale, center, src_dir):
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * self.shift
        src[1, :] = center + src_dir + scale * self.shift
        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        return src


# dst name 
class UpdateDSTMatrix(pr.Processor):
    def __init__(self):
        super(UpdateDSTMatrix, self).__init__()
        self.get_3rd_point = Get3rdPoint()

    def call(self, output_size):
        dst_W = output_size[0]
        dst_H = output_size[1]
        dst_dir = np.array([0, dst_W * -0.5], np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_W * 0.5, dst_H * 0.5]
        dst[1, :] = np.array([dst_W * 0.5, dst_H * 0.5]) + dst_dir
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])
        return dst


# cv2 functions inside the processor
class GetAffineTransform(pr.Processor):
    def __init__(self):
        super(GetAffineTransform, self).__init__()

    def call(self, dst, src):
        transform = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        return transform


class WarpAffine(pr.Processor):
    def __init__(self):
        super(WarpAffine, self).__init__()

    def call(self, image, transform, size_resized):
        image_resized = cv2.warpAffine(image, transform, size_resized)
        return image_resized


class UpSampling2D(pr.Processor):
    def __init__(self, size, interpolation):
        super(UpSampling2D, self).__init__()
        self.size = size
        self.interpolation = interpolation

    def call(self, x):
        x = [tf.keras.layers.UpSampling2D(size=self.size, 
             interpolation=self.interpolation)(each) for each in x]
        return x
         

class UpdateHeatmapAverage(pr.Processor):
    def __init__(self):
        super(UpdateHeatmapAverage, self).__init__()

    def call(self, heatmaps_average, output, indices, num_joints, with_flip=False):
        if not with_flip:
            heatmaps_average += output[:, :, :, :num_joints]

        else:
            temp = output[:, :, :, :num_joints]
            heatmaps_average += tf.gather(temp, indices, axis=-1)
        return heatmaps_average


class IncrementByOne(pr.Processor):
    def __init__(self):
        super(IncrementByOne, self).__init__()

    def call(self, x):
        x += 1
        return x
        

class UpdateTags(pr.Processor):
    def __init__(self, tag_per_joint):
        super(UpdateTags, self).__init__()
        self.tag_per_joint = tag_per_joint
        
    def call(self, tags, output, offset, indices, with_flip=False):
        tags.append(output[:, :, :, offset:])
        if with_flip and self.tag_per_joint:
            tags[-1] = tf.gather(tags[-1], indices, axis=-1)
        return tags
        

class UpdateHeatmaps(pr.Processor):
    def __init__(self):
        super(UpdateHeatmaps, self).__init__()
        
    def call(self, heatmaps, heatmap_average, num_heatmaps):
        heatmaps.append(heatmap_average/num_heatmaps)
        return heatmaps


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


class FlipJointOrder(pr.Processor):
    def __init__(self, with_center):
        super(FlipJointOrder, self).__init__()
        self.with_center = with_center
        
    def call(self):
        if not self.with_center:
            idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        else:
            idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17]
        return idx


class RemoveLastElement(pr.Processor):
    def __init__(self):
        super(RemoveLastElement, self).__init__()

    def call(self, nested_list):
        return [each_list[:, :-1] for each_list in nested_list]

        
        