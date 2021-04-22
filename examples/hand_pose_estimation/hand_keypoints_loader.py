import glob
import numpy as np
import pickle

from paz.abstract import Loader
from paz.backend.image.opencv_image import load_image, resize_image


class HandDataset(Loader):
    def __init__(self, path, split='train', image_size=(256, 256, 3),
                 use_wrist_coordinates=True, flip_to_left=True, crop_size=256,
                 crop_image_after_segmentation=False, sigma=25.0 ):
        super().__init__(path, split, None, 'HandPoseEstimation')
        self.path = path
        self.split = split
        self.image_size = image_size
        self.target_image_size = image_size[0:2]
        self.use_wrist_coordinates = use_wrist_coordinates
        self.kinematic_chain_dict = {0: 'root',
                                     4: 'root', 3: 4, 2: 3, 1: 2,
                                     8: 'root', 7: 8, 6: 7, 5: 6,
                                     12: 'root', 11: 12, 10: 11, 9: 10,
                                     16: 'root', 15: 16, 14: 15, 13: 14,
                                     20: 'root', 19: 20, 18: 19, 17: 18}
        self.kinematic_chain_list = list(self.kinematic_chain_dict.keys())
        self.flip_to_left = flip_to_left
        self.crop_image_hand_localization = crop_image_after_segmentation
        self.crop_size = crop_size
        self.sigma = sigma

    def _load_images(self, image_path):
        image = load_image(image_path)
        hand = resize_image(image, (self.image_size[0], self.image_size[1]))
        return hand

    def _load_keypoints_3D(self, keypoints_3D):
        if not self.use_wrist_coordinates:
            palm_coordinates_left = np.expand_dims(
                0.5 * (keypoints_3D[0, :] + keypoints_3D[12, :]), 0)
            palm_coordinates_right = np.expand_dims(
                0.5 * (keypoints_3D[21, :] + keypoints_3D[33, :]), 0)
            keypoints_3D = np.concat(
                [palm_coordinates_left, keypoints_3D[1:21, :],
                 palm_coordinates_right, keypoints_3D[-20:, :]], 0)

        return keypoints_3D

    def _load_keypoint_2D(self, keypoint_2D):
        if not self.use_wrist_coordinates:
            palm_coordinates_uv_left = np.expand_dims(
                0.5 * (keypoint_2D[0, :] + keypoint_2D[12, :]), 0)
            palm_coordinates_uv_right = np.expand_dims(
                0.5 * (keypoint_2D[21, :] + keypoint_2D[33, :]), 0)
            keypoint_2D = np.concat([palm_coordinates_uv_left,
                                     keypoint_2D[1:21, :],
                                     palm_coordinates_uv_right,
                                     keypoint_2D[-20:, :]], 0)
        return keypoint_2D

    def to_homogeneous_coordinates(self, vector):
        vector = np.reshape(vector, [1, -1, 1])
        vector = np.concat([vector, np.ones((1, 1, 1))], 1)
        return vector

    def _gen_matrix_from_vectors(self, vectors):
        vector_list = [np.reshape(x, [1, 1]) for x in vectors]

        transformation_matrix = np.dynamic_stitch([[0], [1], [2], [3],
                                                   [4], [5], [6], [7],
                                                   [8], [9], [10], [11],
                                                   [12], [13], [14], [15]],
                                                  vector_list)

        transformation_matrix = np.reshape(transformation_matrix,
                                           [4, 4, 1])
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
        return transformation_matrix

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
        background_mask = np.logical_not(hand_mask)
        return np.cast(np.stack([background_mask, hand_mask], 2), np.int32)

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
        num_pixels_left_hand = np.reduce_sum(hand_map_left)
        num_pixels_right_hand = np.reduce_sum(hand_map_right)

        kp_coord_xyz_left = key_point_3D[:21, :]
        kp_coord_xyz_right = key_point_3D[-21:, :]

        dominant_hand = np.logical_and(
            np.cast(np.ones_like(kp_coord_xyz_left), np.bool),
            np.greater(num_pixels_left_hand, num_pixels_right_hand))
        hand_side_keypoints = np.where(dominant_hand, kp_coord_xyz_left,
                                       kp_coord_xyz_right)

        hand_side = np.where(np.greater(num_pixels_left_hand,
                                        num_pixels_right_hand),
                             np.constant(0, dtype=np.int32),
                             np.constant(1, dtype=np.int32))

        hand_side_one_hot = np.one_hot(hand_side, depth=2, on_value=1.0,
                                       off_value=0.0, dtype=np.float32)

        return hand_side_one_hot, hand_side_keypoints, dominant_hand

    def _normalize_keypoints(self, keypoints):
        keypoint_coord_xyz_root = keypoints[0, :]
        keypoint_coord_xyz21_rel = keypoints - keypoint_coord_xyz_root
        index_root_bone_length = np.sqrt(np.reduce_sum(
            np.square(keypoint_coord_xyz21_rel[12, :] -
                      keypoint_coord_xyz21_rel[11, :])))
        keypoint_scale = index_root_bone_length
        keypoint_normed = keypoint_coord_xyz21_rel / index_root_bone_length
        return keypoint_scale, keypoint_normed

    def _get_canonical_transformations(self, keypoints_3D):
        keypoints = np.reshape(keypoints_3D, [-1, 21, 3])

        ROOT_NODE_ID = 0
        ALIGN_NODE_ID = 12
        LAST_NODE_ID = 20

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

        reference_keypoint_z_rotation = resultant_keypoints[:, LAST_NODE_ID, :]
        gamma = np.arctan2(reference_keypoint_z_rotation[:, 2],
                           reference_keypoint_z_rotation[:, 0])
        rotation_matrix_y = self._get_rotation_matrix_y(gamma)
        keypoints_transformed = np.matmul(resultant_keypoints,
                                          rotation_matrix_y)

        final_rotation_matrix = np.matmul(np.matmul(rotation_matrix_z,
                                                    rotation_matrix_x),
                                          rotation_matrix_y)
        return keypoints_transformed, final_rotation_matrix

    def _extract_dominant_hand_visibility(self, keypoint_visibility,
                                          dominant_hand):
        keypoint_visibility_left = keypoint_visibility[:21]
        keypoint_visibility_right = keypoint_visibility[-21:]
        keypoint_visibility_21 = np.where(dominant_hand[:, 0],
                                          keypoint_visibility_left,
                                          keypoint_visibility_right)
        return keypoint_visibility_21

    def _extract_dominant_2D_keypoints(self, keypoint_2D_visibility,
                                       dominant_hand):
        keypoint_visibility_left = keypoint_2D_visibility[:21, :]
        keypoint_visibility_right = keypoint_2D_visibility[-21:, :]
        keypoint_visibility_2D_21 = np.where(dominant_hand[:, :2],
                                             keypoint_visibility_left,
                                             keypoint_visibility_right)
        return keypoint_visibility_2D_21

    def _flip_right_hand(self, canonical_keypoints, flip_right):
        shape = canonical_keypoints.shape()
        expanded = False
        if len(shape) == 2:
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

    def _crop_image_from_coordinates(self, image, crop_location, crop_size,
                                     scale=0.1):
        s = np.shape(image)
        scale = np.reshape(scale, [-1])
        crop_location = np.cast(crop_location, np.float32)
        crop_location = np.reshape(crop_location, [s[0], 2])
        crop_size = np.cast(crop_size, np.float32)

        crop_size_scaled = crop_size / scale
        y1 = crop_location[:, 0] - crop_size_scaled // 2
        y2 = y1 + crop_size_scaled
        x1 = crop_location[:, 1] - crop_size_scaled // 2
        x2 = x1 + crop_size_scaled
        y1 /= s[1]
        y2 /= s[1]
        x1 /= s[2]
        x2 /= s[2]
        boxes = np.stack([y1, x1, y2, x2], -1)

        crop_size = np.cast(np.stack([crop_size, crop_size]), np.int32)
        box_ind = np.range(s[0])
        image_c = np.image.crop_and_resize(np.cast(image, np.float32), boxes,
                                           box_ind, crop_size, name='crop')
        return image_c

    def _crop_image_based_on_segmentation(self, keypoints_2D, keypoints_2D_vis,
                                          image, camera_matrix):
        crop_center = keypoints_2D[12, ::-1]
        crop_center = np.cond(np.reduce_all(np.is_finite(crop_center)),
                              lambda: crop_center,
                              lambda: np.constant([0.0, 0.0]))
        crop_center = np.reshape(crop_center, [2, ])

        keypoint_h = np.boolean_mask(keypoints_2D[:, 1], keypoints_2D_vis)
        keypoint_w = np.boolean_mask(keypoints_2D[:, 0], keypoints_2D_vis)
        kp_coord_hw = np.stack([keypoint_h, keypoint_w], 1)

        # determine size of crop (measure spatial extend of hw coords first)
        min_coordinates = np.maximum(np.reduce_min(kp_coord_hw, 0), 0.0)
        max_coordinates = np.minimum(np.reduce_max(kp_coord_hw, 0),
                                     self.target_image_size)

        # find out larger distance wrt the center of crop
        crop_size_best = 2 * np.maximum(max_coordinates - crop_center,
                                        crop_center - min_coordinates)
        crop_size_best = np.reduce_max(crop_size_best)
        crop_size_best = np.minimum(np.maximum(crop_size_best, 50.0), 500.0)

        # catch problem, when no valid kp available
        crop_size_best = np.cond(np.reduce_all(np.is_finite(crop_size_best)),
                                 lambda: crop_size_best,
                                 lambda: np.constant(200.0))
        crop_size_best = crop_size_best.reshape([])

        # calculate necessary scaling
        scale = np.cast(self.crop_size, np.float32) / crop_size_best
        scale = np.minimum(np.maximum(scale, 1.0), 10.0)

        # Crop image
        img_crop = self._crop_image_from_coordinates(np.expand_dims(image, 0),
                                                     crop_center,
                                                     self.crop_size, scale)

        # Modify uv21 coordinates
        crop_center_float = np.cast(crop_center, np.float32)
        keypoint_uv21_u = (keypoints_2D[:, 0] - crop_center_float[
            1]) * scale + self.crop_size // 2
        keypoint_uv21_v = (keypoints_2D[:, 1] - crop_center_float[
            0]) * scale + self.crop_size // 2
        keypoint_uv21 = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)

        # Modify camera intrinsics
        scale = np.reshape(scale, [1, ])
        scale_matrix = np.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]], [scale, [0.0], [0.0],
                                                           [0.0], scale, [0.0],
                                                           [0.0], [0.0], [1.0]])
        scale_matrix = np.reshape(scale_matrix, [3, 3])

        crop_center_float = np.cast(crop_center, np.float32)
        trans1 = crop_center_float[0] * scale - self.crop_size // 2
        trans2 = crop_center_float[1] * scale - self.crop_size // 2
        trans1 = np.reshape(trans1, [1, ])
        trans2 = np.reshape(trans2, [1, ])
        trans_matrix = np.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]],
                                         [[1.0], [0.0], -trans2,
                                          [0.0], [1.0], -trans1,
                                          [0.0], [0.0], [1.0]])
        trans_matrix = np.reshape(trans_matrix, [3, 3])

        camera_matrix_cropped = np.matmul(trans_matrix, np.matmul(scale_matrix,
                                                                  camera_matrix)
                                          )

        return scale, np.squeeze(img_crop), keypoint_uv21, camera_matrix_cropped

    def _create_multiple_gaussian_map(self, uv_coordinates, scoremap_size,
                                      sigma, valid_vec):
        sigma = np.cast(sigma, np.float32)
        assert len(scoremap_size) == 2
        s = uv_coordinates.shape()
        coords_uv = np.cast(uv_coordinates, np.int32)
        if valid_vec is not None:
            valid_vec = np.cast(valid_vec, np.float32)
            valid_vec = np.squeeze(valid_vec)
            cond_val = np.greater(valid_vec, 0.5)
        else:
            cond_val = np.ones_like(coords_uv[:, 0], dtype=np.float32)
            cond_val = np.greater(cond_val, 0.5)

        cond_1_in = np.logical_and(np.less(coords_uv[:, 0],
                                           scoremap_size[0] - 1),
                                   np.greater(coords_uv[:, 0], 0))
        cond_2_in = np.logical_and(np.less(coords_uv[:, 1],
                                           scoremap_size[1] - 1),
                                   np.greater(coords_uv[:, 1], 0))
        cond_in = np.logical_and(cond_1_in, cond_2_in)
        cond = np.logical_and(cond_val, cond_in)

        coords_uv = np.cast(coords_uv, np.float32)

        # create meshgrid
        x_range = np.expand_dims(np.range(scoremap_size[0]), 1)
        y_range = np.expand_dims(np.range(scoremap_size[1]), 0)

        X = np.cast(np.tile(x_range, [1, scoremap_size[1]]), np.float32)
        Y = np.cast(np.tile(y_range, [scoremap_size[0], 1]), np.float32)

        X.set_shape((scoremap_size[0], scoremap_size[1]))
        Y.set_shape((scoremap_size[0], scoremap_size[1]))

        X = np.expand_dims(X, -1)
        Y = np.expand_dims(Y, -1)

        X_b = np.tile(X, [1, 1, s[0]])
        Y_b = np.tile(Y, [1, 1, s[0]])

        X_b -= coords_uv[:, 0]
        Y_b -= coords_uv[:, 1]

        dist = np.square(X_b) + np.square(Y_b)

        scoremap = np.exp(-dist / np.square(sigma)) * np.cast(cond, np.float32)

        return scoremap

    def _create_score_maps(self, keypoint_2D, keypoint_vis21):
        keypoint_hw21 = np.stack([keypoint_2D[:, 1], keypoint_2D[:, 0]], -1)

        scoremap_size = self.target_image_size

        if self.crop_image_hand_localization:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = self._create_multiple_gaussian_map(keypoint_hw21,
                                                      scoremap_size,
                                                      self.sigma,
                                                      valid_vec=keypoint_vis21)

        return scoremap

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

                sample['hand_side_one_hot'], sample['hand_side_3Dkey_points'], \
                dominant_hand = \
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

                sample['visibile_21_3Dkeypoints'] = \
                    self._extract_dominant_hand_visibility(
                        sample['key_point_visibility'], dominant_hand)

                sample['visibile_21_2Dkeypoints'] = \
                    self._extract_dominant_hand_visibility(
                        sample['key_points_2D'], dominant_hand)

                if self.crop_image_hand_localization:
                    sample['scale'], sample['img_crop'], \
                    sample['visibile_21_2Dkeypoints'], \
                    sample['camera_matrix_cropped'] = \
                        self._crop_image_based_on_segmentation(
                            sample['visibile_21_2Dkeypoints'],
                            sample['visibile_21_3Dkeypoints'], sample['image'],
                            sample['camera_matrix'])

                sample['score_maps'] = self._create_score_maps(
                    sample['visibile_21_3Dkeypoints'],
                    sample['visibile_21_3Dkeypoints'])
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
