import glob
import pickle

import numpy as np

from backend import normalize_keypoints, to_homogeneous_coordinates, \
    get_translation_matrix, one_hot_encode, extract_hand_side, \
    get_canonical_transformations, flip_right_hand, \
    extract_dominant_hand_visibility, extract_dominant_keypoints2D, \
    crop_image_from_coordinates, create_multiple_gaussian_map, \
    get_geometric_entities
from paz.abstract import Loader
from paz.backend.image.opencv_image import load_image, resize_image


class HandPoseLoader(Loader):
    def __init__(self, path, split='train', image_size=(320, 320, 3),
                 use_wrist_coordinates=True, flip_to_left=True, crop_size=256,
                 image_crop=True, sigma=25.0):
        super().__init__(path, split, None, 'HandSegmentation')
        self.path = path
        self.image_size = image_size
        split_to_folder = {'train': 'training', 'val': 'evaluation',
                           'test': 'testing'}
        self.folder = split_to_folder[split]
        self.use_wrist_coordinates = use_wrist_coordinates
        self.kinematic_chain_dict = {0: 'root',
                                     4: 'root', 3: 4, 2: 3, 1: 2,
                                     8: 'root', 7: 8, 6: 7, 5: 6,
                                     12: 'root', 11: 12, 10: 11, 9: 10,
                                     16: 'root', 15: 16, 14: 15, 13: 14,
                                     20: 'root', 19: 20, 18: 19, 17: 18}
        self.kinematic_chain_list = list(self.kinematic_chain_dict.keys())
        self.flip_to_left = flip_to_left
        self.crop_image = image_crop
        self.crop_size = crop_size
        self.sigma = sigma

    def load_images(self, image_path):
        image = load_image(image_path)
        image = resize_image(image, (self.image_size[0], self.image_size[1]))
        return image

    def extract_hand_mask(self, segmentation_label):
        hand_mask = np.greater(segmentation_label, 1)
        background_mask = np.logical_not(hand_mask)
        return np.stack([background_mask, hand_mask], 2)

    def process_keypoints_3D(self, keypoints_3D):
        if not self.use_wrist_coordinates:
            palm_coordinates_left = np.expand_dims(
                0.5 * (keypoints_3D[0, :] + keypoints_3D[12, :]), 0)
            palm_coordinates_right = np.expand_dims(
                0.5 * (keypoints_3D[21, :] + keypoints_3D[33, :]), 0)
            keypoints_3D = np.concatenate(
                [palm_coordinates_left, keypoints_3D[1:21, :],
                 palm_coordinates_right, keypoints_3D[21:43, :]], 0)
        return keypoints_3D

    def process_keypoint_2D(self, keypoint_2D):
        if not self.use_wrist_coordinates:
            palm_coordinates_uv_left = np.expand_dims(
                0.5 * (keypoint_2D[0, :] + keypoint_2D[12, :]), 0)
            palm_coordinates_uv_right = np.expand_dims(
                0.5 * (keypoint_2D[21, :] + keypoint_2D[33, :]), 0)
            keypoint_2D = np.concatenate([palm_coordinates_uv_left,
                                          keypoint_2D[1:21, :],
                                          palm_coordinates_uv_right,
                                          keypoint_2D[21:43, :]], 0)
        return keypoint_2D

    def extract_visibility_mask(self, visibility_mask):
        # calculate palm visibility
        if not self.use_wrist_coordinates:
            palm_vis_left = np.expand_dims(np.logical_or(visibility_mask[0],
                                                         visibility_mask[12]),
                                           0)
            palm_vis_right = np.expand_dims(np.logical_or(visibility_mask[21],
                                                          visibility_mask[33]),
                                            0)
            visibility_mask = np.concatenate(
                [palm_vis_left, visibility_mask[1:21],
                 palm_vis_right, visibility_mask[21:43]], 0)
        return visibility_mask

    def transform_to_relative_frames(self, keypoints_3D):
        keypoints_3D = keypoints_3D.reshape([21, 3])

        tranformations = [None] * len(self.kinematic_chain_list)
        relative_coordinates = [0.0] * len(self.kinematic_chain_list)

        for bone_index in self.kinematic_chain_list:
            parent_key = self.kinematic_chain_dict[bone_index]
            if parent_key == 'root':
                keypoints_residual = to_homogeneous_coordinates(
                    np.expand_dims(keypoints_3D[bone_index, :], 1))
                print(keypoints_residual.shape)

                Translation_matrix = get_translation_matrix(
                    np.zeros_like(keypoints_3D[0, 0]))

                geometric_entities = get_geometric_entities(
                    keypoints_residual, Translation_matrix)
                relative_coordinates[bone_index] = np.stack(
                    geometric_entities[:3], 0)
                tranformations[bone_index] = geometric_entities[3]
            else:
                Transformation_matrix = tranformations[parent_key]
                x_local_parent = np.matmul(
                    Transformation_matrix,
                    to_homogeneous_coordinates(np.expand_dims(
                        keypoints_3D[parent_key, :], 1)))
                x_local_child = np.matmul(
                    Transformation_matrix,
                    to_homogeneous_coordinates(np.expand_dims(
                        keypoints_3D[bone_index, :], 1)))

                # calculate bone vector in local coords
                delta_vec = x_local_child - x_local_parent
                delta_vec = to_homogeneous_coordinates(np.expand_dims(
                    delta_vec[:, :3], 1))

                # get articulation angles from bone vector
                geometric_entities = get_geometric_entities(
                    delta_vec, Transformation_matrix)

                # save results
                relative_coordinates[bone_index] = np.stack(
                    geometric_entities[:3])
                tranformations[bone_index] = geometric_entities[3]

        key_point_relative_frame = np.stack(relative_coordinates, 1)

        return key_point_relative_frame

    def crop_image_based_on_segmentation(self, keypoints_2D, keypoints_2D_vis,
                                         image, camera_matrix):
        crop_center = keypoints_2D[12, ::-1]

        if not np.all(np.isfinite(crop_center)):
            crop_center = np.array([0.0, 0.0])

        crop_center = np.reshape(crop_center, [2, ])

        keypoint_h = keypoints_2D[:, 1][keypoints_2D_vis]
        keypoint_w = keypoints_2D[:, 0][keypoints_2D_vis]
        kp_coord_hw = np.stack([keypoint_h, keypoint_w], 1)

        min_coordinates = np.maximum(np.amin(kp_coord_hw, 0), 0.0)
        max_coordinates = np.minimum(np.amax(kp_coord_hw, 0),
                                     self.image_size[0:2])

        crop_size_best = 2 * np.maximum(max_coordinates - crop_center,
                                        crop_center - min_coordinates)
        crop_size_best = np.amax(crop_size_best)
        crop_size_best = np.minimum(np.maximum(crop_size_best, 50.0), 500.0)

        if not np.isfinite(crop_size_best):
            crop_size_best = 200.0

        scale = self.crop_size / crop_size_best
        scale = np.minimum(np.maximum(scale, 1.0), 10.0)

        # Crop image
        img_crop = crop_image_from_coordinates(image, crop_center,
                                               self.crop_size, scale)

        keypoint_uv21_u = (keypoints_2D[:, 0] - crop_center[1]) * scale + \
                          self.crop_size // 2
        keypoint_uv21_v = (keypoints_2D[:, 1] - crop_center[0]) * scale + \
                          self.crop_size // 2
        keypoint_uv21 = np.stack([keypoint_uv21_u, keypoint_uv21_v], 1)

        # Modify camera intrinsics
        scale = np.reshape(scale, [1, ])
        scale_matrix = np.array([scale, 0.0, 0.0,
                                 0.0, scale, 0.0,
                                 0.0, 0.0, 1.0])
        scale_matrix = np.reshape(scale_matrix, [3, 3])

        trans1 = crop_center[0] * scale - self.crop_size // 2
        trans2 = crop_center[1] * scale - self.crop_size // 2

        trans1 = np.reshape(trans1, [1, ])
        trans2 = np.reshape(trans2, [1, ])

        trans_matrix = np.array([1.0, 0.0, -trans2,
                                 0.0, 1.0, -trans1,
                                 0.0, 0.0, 1.0])

        trans_matrix = np.reshape(trans_matrix, [3, 3])

        camera_matrix_cropped = np.matmul(trans_matrix, np.matmul(scale_matrix,
                                                                  camera_matrix)
                                          )

        return scale, np.squeeze(img_crop), keypoint_uv21, camera_matrix_cropped

    def create_score_maps(self, keypoint_2D, keypoint_vis21):
        keypoint_hw21 = np.stack([keypoint_2D[:, 1], keypoint_2D[:, 0]], -1)

        scoremap_size = self.image_size[0:2]

        if self.crop_image:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = create_multiple_gaussian_map(keypoint_hw21,
                                                scoremap_size,
                                                self.sigma,
                                                valid_vec=keypoint_vis21)

        return scoremap

    def to_list_of_dictionaries(self, hands, segmentation_labels=None,
                                annotations=None):
        dataset = []
        for arg in range(len(hands)):
            sample = dict()
            sample['image'] = self.load_images(hands[arg])
            if segmentation_labels is not None:
                sample['seg_label'] = self.load_images(
                    segmentation_labels[arg])
                sample['hand_mask'] = self.extract_hand_mask(
                    sample['seg_label'])
            if annotations is not None:
                sample['key_points_3D'] = self.process_keypoints_3D(
                    annotations[arg]['xyz'])

                sample['key_points_2D'] = self.process_keypoint_2D(
                    annotations[arg]['uv_vis'][:, :2])

                sample['key_point_visibility'] = self.extract_visibility_mask(
                    annotations[arg]['uv_vis'][:, 2] == 1)

                sample['camera_matrix'] = annotations[arg]['K']

                hand_side, sample['dominant_3D_keypoints'], dominant_hand = \
                    extract_hand_side(sample['seg_label'],
                                      sample['key_points_3D'])

                sample['hand_side_one_hot'] = one_hot_encode(hand_side, 2)

                sample['scale'], sample['normalized_keypoints'] = \
                    normalize_keypoints(sample['dominant_3D_keypoints'])

                sample['keypoints_local_frame'] = np.squeeze(
                    self.transform_to_relative_frames(
                        sample['normalized_keypoints']))

                canonical_keypoints, rotation_matrix = \
                    get_canonical_transformations(
                        sample['keypoints_local_frame'])

                sample['canonical_keypoints'] = flip_right_hand(
                    canonical_keypoints, np.logical_not(self.flip_to_left))

                sample['rotation_matrix'] = np.linalg.pinv(rotation_matrix)

                sample['visibility_21_3Dkeypoints'] = \
                    extract_dominant_hand_visibility(
                        sample['key_point_visibility'], dominant_hand)

                sample['visibile_21_2Dkeypoints'] = \
                    extract_dominant_keypoints2D(sample['key_points_2D'],
                                                  dominant_hand)

                if self.crop_image:
                    sample['scale'], sample['image_crop'], \
                    sample['visibile_21_2Dkeypoints'], \
                    sample['camera_matrix_cropped'] = \
                        self.crop_image_based_on_segmentation(
                            sample['visibile_21_2Dkeypoints'],
                            sample['visibility_21_3Dkeypoints'],
                            sample['image'],
                            sample['camera_matrix'])

                sample['score_maps'] = self.create_score_maps(
                    sample['visibile_21_2Dkeypoints'],
                    sample['visibility_21_3Dkeypoints'])

            dataset.append(sample)
        return dataset

    def _load_annotation(self, label_path):
        with open(label_path, 'rb') as file:
            annotations_all = pickle.load(file)
        return annotations_all

    def load_data(self):
        images = sorted(glob.glob(self.path + self.folder + '/color/*.png'))
        if self.split == 'train':
            segmentation_labels = sorted(glob.glob(self.path + self.folder +
                                                   '/mask/*.png'))
            annotations = self._load_annotation(self.path + self.folder +
                                                '/anno_training.pickle')
            dataset = self.to_list_of_dictionaries(images, segmentation_labels,
                                                   annotations)

        elif self.split == 'val':
            segmentation_labels = sorted(glob.glob(self.path + self.folder +
                                                   '/mask/*.png'))
            annotations = self._load_annotation(self.path + self.folder +
                                                '/anno_evaluation.pickle')
            dataset = self.to_list_of_dictionaries(images, segmentation_labels,
                                                   annotations)

        else:
            dataset = self.to_list_of_dictionaries(images, None, None)

        return dataset


if __name__ == '__main__':
    dataset = HandPoseLoader(
        '/home/dfki.uni-bremen.de/jbandlamudi/DFKI_Work/RHD_published_v2/')
    dataset = dataset.load_data()
