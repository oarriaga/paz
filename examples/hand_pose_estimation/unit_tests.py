from backend import one_hot_encode, normalize_keypoints, \
    to_homogeneous_coordinates, get_translation_matrix, \
    build_rotation_matrix_x, build_rotation_matrix_y, build_rotation_matrix_z, \
    get_affine_matrix, extract_hand_side, get_canonical_transformations

from data_loaders import HandPoseLoader

data_loader = HandPoseLoader(
    '/media/jarvis/CommonFiles/5th_Semester/DFKI_Work/RHD_published_v2/')

import numpy as np

np.random.seed(0)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _print_pass():
    print(f"{bcolors.OKGREEN}PASSED: All TESTS PASSED {bcolors.ENDC}")


def _print_fail():
    print(f"{bcolors.FAIL}FAILED: TEST FAILED {bcolors.ENDC}")


def test_image_loading(image_path):
    image = data_loader.load_images(image_path)
    assert image.shape == data_loader.image_size, _print_fail()
    _print_pass()


def test_segmentation_map_loading(segmentation_path):
    segmentation_mask = data_loader.load_images(segmentation_path)
    assert segmentation_mask.shape == data_loader.image_size, _print_fail()
    _print_pass()


def test_one_hot_encode():
    one_hot_vector = one_hot_encode(1, 2)
    assert type(one_hot_vector).__module__ == np.__name__, _print_fail()
    assert one_hot_vector.all() == np.array([0, 1]).all(), _print_fail()
    assert one_hot_encode(0, 2).all() == np.array([1, 0]).all(), _print_fail()
    _print_pass()


def test_normalize_keypoints():
    test_array = np.array([[0., 0., 0.], [1., 1., 1.], [1., 1., 1.],
                           [2., 2., 2.], [2., 2., 2.], [3., 3., 3.],
                           [3., 3., 3.], [4., 4., 4.], [5., 5., 5.],
                           [5., 5., 5.], [6., 6., 6.], [6., 6., 6.],
                           [7., 7., 7.], [8., 8., 8.], [8., 8., 8.],
                           [9., 9., 9.], [9., 9., 9.], [10., 10., 10.],
                           [10., 10., 10.], [11., 11., 11.], [12., 12., 12.]])
    keypoints3D = np.random.rand(21, 3)
    keypoint_scale, keypoint_normalized = normalize_keypoints(keypoints3D)
    assert round(keypoint_scale, 2) == 0.68, _print_fail()
    assert keypoints3D.shape == keypoint_normalized.shape, _print_fail()
    assert keypoint_normalized.round().all() == test_array.all(), _print_fail()
    _print_pass()


def test_to_homogeneous():
    vector_shape = (1, 3)
    keypoint = np.zeros(vector_shape)
    homogeneous_keypoint = to_homogeneous_coordinates(keypoint)
    assert homogeneous_keypoint[-1] == 1, _print_fail()
    assert homogeneous_keypoint.shape == (vector_shape[1] + 1,), _print_fail()
    _print_pass()


def test_to_translation():
    translation_matrix = get_translation_matrix(1)
    assert translation_matrix.shape == (1, 4, 4), _print_fail()
    assert translation_matrix[-1].all() == np.array([0, 0, 0, 1]).all(), \
        _print_fail()
    _print_pass()


def test_rotation_matrix_x():
    rotation_matrix_test = np.array([[1.0000000, 0.0000000, 0.0000000],
                                     [0.0000000, 0.8668, 0.5],
                                     [0.0000000, -0.5, 0.8668]])
    rotation_matrix = build_rotation_matrix_x(np.deg2rad(30))
    assert rotation_matrix.shape == rotation_matrix_test.shape, _print_fail()
    assert np.round(np.linalg.det(rotation_matrix)) == 1.0, _print_fail()
    assert np.round(np.linalg.inv(rotation_matrix)).all() == \
           np.round(np.transpose(rotation_matrix)).all(), _print_fail()
    assert rotation_matrix_test.round().all() == \
           rotation_matrix.round().all(), _print_fail()
    _print_pass()


def test_rotation_matrix_y():
    rotation_matrix_test = np.array([[0.8660254, 0.0000000, 0.5000000],
                                     [0.0000000, 1.0000000, 0.0000000],
                                     [-0.5000000, 0.0000000, 0.8660254]])
    rotation_matrix = build_rotation_matrix_y(np.deg2rad(30))
    assert rotation_matrix.shape == rotation_matrix_test.shape, _print_fail()
    assert np.round(np.linalg.det(rotation_matrix)) == 1.0, _print_fail()
    assert np.round(np.linalg.inv(rotation_matrix)).all() == \
           np.round(np.transpose(rotation_matrix)).all(), _print_fail()
    assert rotation_matrix_test.round().all() == \
           rotation_matrix.round().all(), _print_fail()
    _print_pass()


def test_rotation_matrix_z():
    rotation_matrix_test = np.array([[0.8660254, -0.5000000, 0.0000000],
                                     [0.5000000, 0.8660254, 0.0000000],
                                     [0.0000000, 0.0000000, 1.0000000]])
    rotation_matrix = build_rotation_matrix_z(np.deg2rad(30))
    assert rotation_matrix.shape == rotation_matrix_test.shape, _print_fail()
    assert np.round(np.linalg.det(rotation_matrix)) == 1.0, _print_fail()
    assert np.round(np.linalg.inv(rotation_matrix)).all() == \
           np.round(np.transpose(rotation_matrix)).all(), _print_fail()
    assert rotation_matrix_test.round().all() == \
           rotation_matrix.round().all(), _print_fail()
    _print_pass()


def test_get_affine_matrix():
    rotation_matrix = build_rotation_matrix_x(np.deg2rad(30))
    affine_rotation_matrix = get_affine_matrix(rotation_matrix)
    assert affine_rotation_matrix.shape == (4, 4), _print_fail()
    assert affine_rotation_matrix[-1].all() == np.array([0, 0, 0, 1]).all(), \
        _print_fail()
    _print_pass()


def test_hand_side_extraction(segmentation_path, label_path):
    segmentation_mask = data_loader.load_images(segmentation_path)
    annotations_all = data_loader._load_annotation(label_path)
    keypoints3D = data_loader.process_keypoints_3D(annotations_all[11]['xyz'])
    hand_side, hand_side_keypoints, dominant_hand_keypoints = \
        extract_hand_side(segmentation_mask, keypoints3D)

    assert type(hand_side).__module__ == np.__name__, _print_fail()
    assert hand_side == np.array([0])
    assert hand_side_keypoints.shape == (21, 3), _print_fail()
    assert dominant_hand_keypoints.shape == (21, 3), _print_fail()
    _print_pass()


def test_canonical_transformations(label_path):
    annotations_all = data_loader._load_annotation(label_path)
    keypoints3D = data_loader.process_keypoints_3D(annotations_all[11]['xyz'])
    transformed_keypoints, rotation_matrix = get_canonical_transformations(
        keypoints3D.T)

    assert transformed_keypoints.shape == (42, 3), _print_fail()
    assert rotation_matrix.shape == (3, 3), _print_fail()
    _print_pass()


if __name__ == '__main__':
    test_canonical_transformations(
        '/media/jarvis/CommonFiles/5th_Semester/DFKI_Work/RHD_published_v2/training/anno_training.pickle')
