import glob
import numpy as np
import pickle

from paz.abstract import Loader
from paz.backend.image.opencv_image import load_image, resize_image


class HandDataset(Loader):
    def __init__(self, path, split='training', image_size=(256, 256, 3),
                 use_wrist_coordinates=True, flip_to_left=True):
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
        self.flip_to_left = flip_to_left

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

    def to_homogeneous_coordinates(self, vector):
        batch_size = vector.shape[0]
        vector = np.reshape(vector, [batch_size, -1, 1])
        vector = np.concat([vector, np.ones((batch_size, 1, 1))], 1)
        return vector

    def _gen_matrix_from_vectors(self, vectors):
        batch_size = vectors.shape[0]
        vector_list = [np.reshape(x, [1, batch_size]) for x in vectors]

        transformation_matrix = np.dynamic_stitch([[0], [1], [2], [3],
                                                   [4], [5], [6], [7],
                                                   [8], [9], [10], [11],
                                                   [12], [13], [14], [15]],
                                                  vector_list)

        transformation_matrix = np.reshape(transformation_matrix,
                                           [4, 4, batch_size])
        transformation_matrix = np.transpose(transformation_matrix, [2, 0, 1])

        return transformation_matrix

    def _get_transformation_matrix_y(self, angle):
        ones, zeros = np.ones_like(angle), np.zeros_like(angle, dtype=np.float)
        rotation_matrix_y = self._gen_matrix_from_vectors(
            np.array([np.cos(angle), zeros, np.sin(angle), zeros,
                      zeros, ones, zeros, zeros,
                      -np.sin(angle), zeros, np.cos(angle), zeros,
                      zeros, zeros, zeros, ones]))
        return rotation_matrix_y

    def _get_transformation_matrix_x(self, angle):
        ones, zeros = np.ones_like(angle), np.zeros_like(angle, dtype=np.float)
        rotation_matrix_x = self._gen_matrix_from_vectors(
            np.array([ones, zeros, zeros, zeros,
                      zeros, np.cos(angle), -np.sin(angle), zeros,
                      zeros, np.sin(angle), np.cos(angle), zeros,
                      zeros, zeros, zeros, ones]))
        return rotation_matrix_x

    def _get_rotation_matrix_x(self, angle):
        ones, zeros = np.ones_like(angle), np.zeros_like(angle, dtype=np.float)
        rotation_matrix_x = self._gen_matrix_from_vectors(
            np.array([ones, zeros, zeros,
                      zeros, np.cos(angle), np.sin(angle),
                      zeros, -np.sin(angle), np.cos(angle)]))
        return rotation_matrix_x

    def _get_rotation_matrix_y(self, angle):
        ones, zeros = np.ones_like(angle), np.zeros_like(angle, dtype=np.float)
        rotation_matrix_x = self._gen_matrix_from_vectors(
            np.array([np.cos(angle), zeros, -np.sin(angle),
                      zeros, ones, zeros,
                      np.sin(angle), zeros, np.cos(angle)]))
        return rotation_matrix_x

    def _get_rotation_matrix_z(self, angle):
        ones, zeros = np.ones_like(angle), np.zeros_like(angle, dtype=np.float)
        rotation_matrix_x = self._gen_matrix_from_vectors(
            np.array([np.cos(angle), np.sin(angle), zeros,
                      -np.sin(angle), np.cos(angle), zeros,
                      zeros, zeros, ones]))
        return rotation_matrix_x

    def _get_translation_matrix(self, translation_vector):
        ones, zeros = np.ones_like(translation_vector), np.zeros_like(
            translation_vector, dtype=np.float)
        rotation_matrix_x = self._gen_matrix_from_vectors(
            np.array([ones, zeros, zeros, zeros,
                      zeros, ones, zeros, zeros,
                      zeros, zeros, ones, translation_vector,
                      zeros, zeros, zeros, ones]))
        return rotation_matrix_x

    def _get_geometric_entities(self, vector, transformation_matrix):
        length_from_origin = np.sqrt(
            vector[:, 0, 0] ** 2 + vector[:, 1, 0] ** 2 +
            vector[:, 2, 0] ** 2)
        gamma = np.arctan2(vector[:, 0, 0], vector[:, 2, 0])

        matrix_after_y_rotation = np.matmul(self._get_transformation_matrix_y(
            -gamma), vector)
        alpha = np.arctan2(-matrix_after_y_rotation[:, 1, 0],
                           matrix_after_y_rotation[:, 2, 0])
        matrix_after_x_rotation = np.matmul(self._get_translation_matrix(
            -length_from_origin), np.matmul(self._get_transformation_matrix_x(
            -alpha), self._get_transformation_matrix_y(-gamma)))

        final_transformation_matrix = np.matmul(matrix_after_x_rotation,
                                                matrix_after_y_rotation)

        # make them all batched scalars
        length_from_origin = np.reshape(length_from_origin, [-1])
        alpha = np.reshape(alpha, [-1])
        gamma = np.reshape(gamma, [-1])
        return length_from_origin, alpha, gamma, final_transformation_matrix

    def _get_homogeneous_transformation_matrix(self, homogeneous_coords):
        ones, zeros = np.ones_like(homogeneous_coords), np.zeros_like(
            homogeneous_coords, dtype=np.float)
        transformation_matrix = self._gen_matrix_from_vectors(
            np.array([ones, zeros, zeros, zeros, zeros, ones, zeros, zeros,
                      zeros, zeros, ones, homogeneous_coords, zeros, zeros,
                      zeros, ones]))

    def transform_to_relative_frames(self, keypoints_3D):
        keypoints_3D = keypoints_3D.reshape([-1, 21, 3])
        tranformations = [None] * len(self.kinematic_chain_list)
        relative_coordinates = [0.0] * len(self.kinematic_chain_list)
        for bone_index in self.kinematic_chain_list:
            parent_key = self.kinematic_chain_dict[bone_index]
            if parent_key == 'root':
                keypoints_residual = self.to_homogeneous_coordinates(
                    np.expand_dims(keypoints_3D[:, bone_index, :], 1))
                Translation_matrix = self._get_translation_matrix(np.zeros_like(
                    keypoints_3D[:, 0, 0]))
                geometric_entities = self._get_geometric_entities(
                    keypoints_residual, Translation_matrix)
                relative_coordinates[bone_index] = np.stack(
                    geometric_entities[:3], 1)
                tranformations[bone_index] = geometric_entities[3]
            else:
                Transformation_matrix = tranformations[parent_key]
                x_local_parent = np.matmul(
                    Transformation_matrix,
                    self.to_homogeneous_coordinates(np.expand_dims(
                        keypoints_3D[:, parent_key, :], 1)))
                x_local_child = np.matmul(
                    Transformation_matrix,
                    self.to_homogeneous_coordinates(np.expand_dims(
                        keypoints_3D[:, bone_index, :], 1)))

                # calculate bone vector in local coords
                delta_vec = x_local_child - x_local_parent
                delta_vec = self.to_homogeneous_coordinates(np.expand_dims(
                    delta_vec[:, :3, :], 1))

                # get articulation angles from bone vector
                geometric_entities = self._get_geometric_entities(
                    delta_vec, Transformation_matrix)

                # save results
                relative_coordinates[bone_index] = np.stack(
                    geometric_entities[:3], 1)
                tranformations[bone_index] = geometric_entities[3]

        key_point_relative_frame = np.stack(relative_coordinates, 1)

        return key_point_relative_frame

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

    def _get_canonical_transformations(self, keypoints_3D):
        keypoints = np.reshape(keypoints_3D, [-1, 21, 3])

        ROOT_NODE_ID = 0
        ALIGN_NODE_ID = 12
        ROT_NODE_ID = 20

        translation_reference = np.expand_dims(keypoints[:, ROOT_NODE_ID, :], 1)
        translated_keypoints = keypoints - translation_reference

        alignment_keypoint = translated_keypoints[:, ALIGN_NODE_ID, :]

        alpha = np.arctan2(alignment_keypoint[:, 0], alignment_keypoint[:, 1])
        rotation_matrix_z = self._get_rotation_matrix_z(alpha)
        resultant_keypoints = np.matmul(translated_keypoints, rotation_matrix_z)

        reference_keypoint_z_rotation = resultant_keypoints[:, ALIGN_NODE_ID, :]
        beta = -np.arctan2(reference_keypoint_z_rotation[:, 2],
                           reference_keypoint_z_rotation[:, 1])
        rotation_matrix_x = self._get_rotation_matrix_x(beta + 3.14159)
        resultant_keypoints = np.matmul(resultant_keypoints, rotation_matrix_x)

        reference_keypoint_z_rotation = resultant_keypoints[:, ROT_NODE_ID, :]
        gamma = np.arctan2(reference_keypoint_z_rotation[:, 2],
                           reference_keypoint_z_rotation[:, 0])
        rotation_matrix_y = self._get_rotation_matrix_y(gamma)
        keypoints_transformed = np.matmul(resultant_keypoints,
                                          rotation_matrix_y)

        final_rotation_matrix = np.matmul(np.matmul(rotation_matrix_z,
                                                    rotation_matrix_x),
                                          rotation_matrix_y)
        return keypoints_transformed, final_rotation_matrix

    def _flip_right_hand(self, canonical_keypoints, flip_right):
        shape = canonical_keypoints.shape()
        expanded = False
        if len(shape)==2:
            canonical_keypoints = np.expand_dims(canonical_keypoints, 0)
            flip_right = np.expand_dims(flip_right, 0)
            expanded = True
        canonical_keypoints_mirrored = np.stack(
            [canonical_keypoints[:, :, 0], canonical_keypoints[:, :, 1],
             -canonical_keypoints[:, :, 2]], -1)

        canonical_keypoints_left = np.where(flip_right,
                                             canonical_keypoints_mirrored,
                                             canonical_keypoints)
        if expanded:
            canonical_keypoints_left = np.squeeze(canonical_keypoints_left,
                                                   [0])
        return canonical_keypoints_left

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
                sample['keypoint_scale'], sample['normalized_keypoints'] = \
                    self._normalize_keypoints(sample['hand_side_3Dkey_points'])
                sample['keypoints_local_frame'] = np.squeeze(
                    self.transform_to_relative_frames(
                        sample['normalized_keypoints']))
                canonical_keypoints, rotation_matrix = \
                    self._get_canonical_transformations(
                        sample['keypoints_local_frame'])
                canonical_keypoints, rotation_matrix = \
                    np.squeeze(canonical_keypoints), np.squeeze(rotation_matrix)
                canonical_keypoints = self._flip_right_hand(
                    canonical_keypoints, np.logical_not(self.flip_to_left))
                sample['canonical_keypoints'] = canonical_keypoints
                sample['rotation_matrix'] = np.matrix_inverse(rotation_matrix)

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
