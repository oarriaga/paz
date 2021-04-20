import glob
import numpy as np
import pickle

from paz.abstract import Loader
from paz.backend.image.opencv_image import load_image, resize_image


class HandDataset(Loader):
    def __init__(self, path, split='training', image_size=(256, 256, 3),
                 use_wrist_coordinates=True, ):
        super().__init__(path, split, None, 'HandPoseEstimation')
        self.path = path
        self.split = split
        self.image_size = image_size
        self.use_wrist_coordinates = use_wrist_coordinates
        self.kinematic_chain_dict = {0: 'root',
                                     4: 'root', 3: 4, 2: 3, 1: 2,
                                     8: 'root', 7: 8, 6: 7, 5: 6,
                                     12: 'root', 11: 12, 10: 11, 9: 10,
                                     16: 'root', 15: 16, 14: 15, 13: 14,
                                     20: 'root', 19: 20, 18: 19, 17: 18}
        self.kinematic_chain_list = list(self.kinematic_chain_dict.keys())

    def _load_images(self, image_path):
        image = load_image(image_path)
        hand = resize_image(image, (self.image_size[0], self.image_size[1]))
        return hand

    def _load_keypoints_3D(self, keypoints_3D):
        if not self.use_wrist_coordinates:
            palm_coord_left = np.expand_dims(
                0.5 * (keypoints_3D[0, :] + keypoints_3D[12, :]), 0)
            palm_coord_right = np.expand_dims(
                0.5 * (keypoints_3D[21, :] + keypoints_3D[33, :]), 0)
            keypoints_3D = np.concat(
                [palm_coord_left, keypoints_3D[1:21, :], palm_coord_right,
                 keypoints_3D[-20:, :]], 0)

        return keypoints_3D

    def _load_keypoint_2D(self, keypoint_2D):
        if not self.use_wrist_coordinates:
            palm_coord_uv_left = np.expand_dims(
                0.5 * (keypoint_2D[0, :] + keypoint_2D[12, :]), 0)
            palm_coord_uv_right = np.expand_dims(
                0.5 * (keypoint_2D[21, :] + keypoint_2D[33, :]), 0)
            keypoint_2D = np.concat([palm_coord_uv_left, keypoint_2D[1:21, :],
                                     palm_coord_uv_right, keypoint_2D[-20:, :]],
                                    0)
        return keypoint_2D

    def _extract_hand_mask(self, segmentation_label):
        hand_mask = np.greater(segmentation_label, 1)
        bg_mask = np.logical_not(hand_mask)
        return np.cast(np.stack([bg_mask, hand_mask], 2), np.int32)

    def _extract_visibility_mask(self, visibility_mask):
        visibility_mask = np.cast(visibility_mask, np.bool)

        # calculate palm visibility
        if not self.use_wrist_coordinates:
            palm_vis_left = np.expand_dims(np.logical_or(visibility_mask[0],
                                                         visibility_mask[12]),
                                           0)
            palm_vis_right = np.expand_dims(np.logical_or(visibility_mask[21],
                                                          visibility_mask[33]),
                                            0)
            visibility_mask = np.concat([palm_vis_left, visibility_mask[1:21],
                                         palm_vis_right, visibility_mask[-20:]],
                                        0)
        return visibility_mask

    def _extract_hand_side(self, hand_parts_mask, key_point_3D):
        one_map, zero_map = np.ones_like(hand_parts_mask), np.zeros_like(
            hand_parts_mask)
        raw_mask_left = np.logical_and(np.greater(hand_parts_mask, one_map),
                                       np.less(hand_parts_mask, one_map * 18))
        raw_mask_right = np.greater(hand_parts_mask, one_map * 17)
        hand_map_left = np.where(raw_mask_left, one_map, zero_map)
        hand_map_right = np.where(raw_mask_right, one_map, zero_map)
        num_px_left_hand = np.reduce_sum(hand_map_left)
        num_px_right_hand = np.reduce_sum(hand_map_right)

        # PRODUCE the 21 subset using the segmentation masks
        # We only deal with the more prominent hand for each frame
        # and discard the second set of keypoints
        kp_coord_xyz_left = key_point_3D[:21, :]
        kp_coord_xyz_right = key_point_3D[-21:, :]

        cond_left = np.logical_and(
            np.cast(np.ones_like(kp_coord_xyz_left), np.bool),
            np.greater(num_px_left_hand, num_px_right_hand))
        hand_side_keypoints = np.where(cond_left, kp_coord_xyz_left,
                                       kp_coord_xyz_right)

        hand_side = np.where(np.greater(num_px_left_hand, num_px_right_hand),
                             np.constant(0, dtype=np.int32),
                             np.constant(1, dtype=np.int32))
        hand_side_one_hot = np.one_hot(hand_side, depth=2, on_value=1.0,
                                       off_value=0.0, dtype=np.float32)

        return hand_side_one_hot, hand_side_keypoints

    def _normalize_keypoints(self, keypoints):
        kp_coord_xyz_root = keypoints[0, :]
        kp_coord_xyz21_rel = keypoints - kp_coord_xyz_root
        index_root_bone_length = np.sqrt(np.reduce_sum(
            np.square(kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])))
        keypoint_scale = index_root_bone_length
        keypoint_normed = kp_coord_xyz21_rel / index_root_bone_length
        return keypoint_scale, keypoint_normed

    def _to_list_of_dictionaries(self, hands, segmentation_labels=None,
                                 annotations=None):
        dataset = []
        for arg in range(len(hands)):
            sample = dict()
            sample['image'] = self._load_images(hands[arg])
            if segmentation_labels is not None:
                sample['seg_label'] = self._load_images(
                    segmentation_labels[arg])
                sample['hand_mask'] = self._extract_hand_mask(
                    sample['seg_label'])
            if annotations is not None:
                sample['key_points_3D'] = self._load_keypoints_3D(
                    annotations[arg]['xyz'])
                sample['key_points_2D'] = self._load_keypoint_2D(
                    annotations[arg]['uv_vis'][:, :2])
                sample['key_point_visibility'] = self._extract_visibility_mask(
                    annotations[arg]['uv_vis'][:, 2] == 1)
                sample['camera_matrix'] = annotations[arg]['K']
                sample['hand_side_one_hot'], sample['hand_side_3Dkey_points'] = \
                    self._extract_hand_side(sample['hand_mask'],
                                            sample['key_points_3D'])
            dataset.append(sample)
        return dataset

    def _load_annotation(self, label_path):
        with open(label_path, 'rb') as file:
            annotations_all = pickle.load(file)
        return annotations_all

    def load_data(self):
        images = sorted(glob.glob(self.path + self.split + '/color/*.png'))
        if self.split == 'training':
            segmentation_labels = sorted(glob.glob(self.path + self.split +
                                                   '/mask/*.png'))
            annotations = self._load_annotation(self.path + self.split +
                                                '/anno_training.pickle')
            dataset = self._to_list_of_dictionaries(images, segmentation_labels,
                                                    annotations)
        else:
            dataset = self._to_list_of_dictionaries(images, None, None)
        return dataset


if __name__ == '__main__':
    path = 'dataset/'
    split = 'training'
    data_manager = HandDataset(path, split)
    dataset = data_manager.load_data()
