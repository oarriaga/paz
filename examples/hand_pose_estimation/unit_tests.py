from backend import normalize_keypoints
from backend import to_homogeneous_coordinates, build_translation_matrix_SE3
from backend import build_rotation_matrix_x, build_rotation_matrix_y
from backend import build_rotation_matrix_z, build_affine_matrix
from backend import get_canonical_transformations, get_hand_side_and_keypooints

from paz.backend.boxes import to_one_hot

from hand_keypoints_loader import RenderedHandLoader

data_loader = RenderedHandLoader(
    '/media/jarvis/CommonFiles/5th_Semester/DFKI_Work/RHD_published_v2/')

from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
import numpy as np
from pipelines import preprocess_image, PostprocessSegmentation, \
    Process2DKeypoints
from paz.backend.image.opencv_image import load_image
from backend import create_multiple_gaussian_map
from processors import ExtractKeypoints

np.random.seed(0)

use_pretrained = True
HandSegNet = HandSegmentationNet(load_pretrained=use_pretrained)
HandPoseNet = PoseNet()
HandPosePriorNet = PosePriorNet()
HandViewPointNet = ViewPointNet()


def test_image_loading(image_path):
    image = data_loader.load_images(image_path)
    assert image.shape == data_loader.image_size


def test_segmentation_map_loading(segmentation_path):
    segmentation_mask = data_loader.load_images(segmentation_path)
    assert segmentation_mask.shape == data_loader.image_size


def test_one_hot_encode():
    one_hot_vector = to_one_hot(1, 2)
    assert type(one_hot_vector).__module__ == np.__name__
    assert one_hot_vector.all() == np.array([0, 1]).all()
    assert to_one_hot(0, 2).all() == np.array([1, 0]).all()


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
    assert round(keypoint_scale, 2) == 0.68
    assert keypoints3D.shape == keypoint_normalized.shape
    assert keypoint_normalized.round().all() == test_array.all()


def test_to_homogeneous():
    vector_shape = (1, 3)
    keypoint = np.zeros(vector_shape)
    homogeneous_keypoint = to_homogeneous_coordinates(keypoint)
    assert homogeneous_keypoint[-1] == 1
    assert homogeneous_keypoint.shape == (vector_shape[1] + 1,)


def test_to_translation():
    translation_matrix = build_translation_matrix_SE3(1)
    assert translation_matrix.shape == (1, 4, 4)
    assert translation_matrix[-1].all() == np.array([0, 0, 0, 1]).all()


def test_rotation_matrix_x():
    rotation_matrix_test = np.array([[1.0000000, 0.0000000, 0.0000000],
                                     [0.0000000, 0.8668, 0.5],
                                     [0.0000000, -0.5, 0.8668]])
    rotation_matrix = build_rotation_matrix_x(np.deg2rad(30))
    assert rotation_matrix.shape == rotation_matrix_test.shape
    assert np.round(np.linalg.det(rotation_matrix)) == 1.0
    assert np.round(np.linalg.inv(rotation_matrix)).all() == \
           np.round(np.transpose(rotation_matrix)).all()
    assert rotation_matrix_test.round().all() == \
           rotation_matrix.round().all()


def test_rotation_matrix_y():
    rotation_matrix_test = np.array([[0.8660254, 0.0000000, 0.5000000],
                                     [0.0000000, 1.0000000, 0.0000000],
                                     [-0.5000000, 0.0000000, 0.8660254]])
    rotation_matrix = build_rotation_matrix_y(np.deg2rad(30))
    assert rotation_matrix.shape == rotation_matrix_test.shape
    assert np.round(np.linalg.det(rotation_matrix)) == 1.0
    assert np.round(np.linalg.inv(rotation_matrix)).all() == \
           np.round(np.transpose(rotation_matrix)).all()
    assert rotation_matrix_test.round().all() == \
           rotation_matrix.round().all()


def test_rotation_matrix_z():
    rotation_matrix_test = np.array([[0.8660254, -0.5000000, 0.0000000],
                                     [0.5000000, 0.8660254, 0.0000000],
                                     [0.0000000, 0.0000000, 1.0000000]])
    rotation_matrix = build_rotation_matrix_z(np.deg2rad(30))
    assert rotation_matrix.shape == rotation_matrix_test.shape
    assert np.round(np.linalg.det(rotation_matrix)) == 1.0
    assert np.round(np.linalg.inv(rotation_matrix)).all() == \
           np.round(np.transpose(rotation_matrix)).all()
    assert rotation_matrix_test.round().all() == \
           rotation_matrix.round().all()


def test_get_affine_matrix():
    rotation_matrix = build_rotation_matrix_x(np.deg2rad(30))
    affine_rotation_matrix = build_affine_matrix(rotation_matrix)
    assert affine_rotation_matrix.shape == (4, 4)
    assert affine_rotation_matrix[-1].all() == np.array([0, 0, 0, 1]).all()


def test_hand_side_extraction(segmentation_path, label_path):
    segmentation_mask = data_loader.load_images(segmentation_path)
    annotations_all = data_loader._load_annotation(label_path)
    keypoints3D = data_loader.process_keypoints_3D(annotations_all[11]['xyz'])
    hand_side, hand_side_keypoints, dominant_hand_keypoints = \
        get_hand_side_and_keypooints(segmentation_mask, keypoints3D)

    assert type(hand_side).__module__ == np.__name__
    assert hand_side == np.array([0])
    assert hand_side_keypoints.shape == (21, 3)
    assert dominant_hand_keypoints.shape == (21, 3)


def test_canonical_transformations(label_path):
    annotations_all = data_loader._load_annotation(label_path)
    keypoints3D = data_loader.process_keypoints_3D(annotations_all[11]['xyz'])
    transformed_keypoints, rotation_matrix = get_canonical_transformations(
        keypoints3D.T)

    assert transformed_keypoints.shape == (42, 3)
    assert rotation_matrix.shape == (3, 3)


def test_preprocess_image():
    preprocess_pipeline = preprocess_image()
    image = load_image('./sample.jpg')
    processed_image = preprocess_pipeline(image)

    assert len(processed_image.shape) == 4
    assert processed_image.shape == (1, 320, 320, 3)


def test_segmentation_postprocess():
    preprocess_pipeline = preprocess_image()
    image = load_image('./sample.jpg')
    processed_image = preprocess_pipeline(image)

    localization_pipeline = PostprocessSegmentation(HandSegNet)
    localization_output = localization_pipeline(processed_image)

    assert len(localization_output) == 5
    assert localization_output[0].shape == (1, 256, 256, 3)
    assert localization_output[1].shape == (1, 320, 320, 1)
    assert localization_output[2].shape == (1, 2)
    assert localization_output[3].shape == (1, 2, 2)
    assert localization_output[4].shape == (1, 1)


def test_keypoints2D_process():
    preprocess_pipeline = preprocess_image()
    image = load_image('./sample.jpg')
    processed_image = preprocess_pipeline(image)

    localization_pipeline = PostprocessSegmentation(HandSegNet)
    localization_output = localization_pipeline(processed_image)

    keypoints_pipeline = Process2DKeypoints(HandPoseNet)
    score_maps_dict = keypoints_pipeline(np.squeeze(localization_output[0],
                                                    axis=0))
    score_maps = score_maps_dict['score_maps']

    assert score_maps.shape == (1, 32, 32, 21)
    assert len(score_maps) == 1


def test_extract_keypoints2D():
    uv_coordinates = np.array([[0, 0], [1, 1]])
    uv_coordinates = np.expand_dims(uv_coordinates, axis=0)

    gaussian_maps = create_multiple_gaussian_map(uv_coordinates, (256, 256),
                                                 sigma=0.1, valid_vec=None)
    gaussian_maps = np.expand_dims(gaussian_maps, axis=0)
    keypoints_extraction_pipeline = ExtractKeypoints()
    keypoints2D = keypoints_extraction_pipeline(gaussian_maps)

    assert keypoints2D[0] == [0, 0]


if __name__ == '__main__':
    test_canonical_transformations(
        '/media/jarvis/CommonFiles/5th_Semester/DFKI_Work/RHD_published_v2/training/anno_training.pickle')
