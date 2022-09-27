from backend_SE3 import build_rotation_matrix_x, build_rotation_matrix_y
from backend_SE3 import build_rotation_matrix_z, build_affine_matrix
from backend_SE3 import rotation_from_axis_angles
from backend_SE3 import to_homogeneous_coordinates, build_translation_matrix_SE3

from backend_keypoints import canonical_transformations_on_keypoints
from backend_keypoints import get_hand_side_and_keypooints
from backend_keypoints import keypoints_to_palm_coordinates
from backend_keypoints import normalize_keypoints, extract_hand_side_keypoints
from RHDv2 import LEFT_WRIST
from RHDv2 import RIGHT_WRIST
from hand_keypoints_loader import RenderedHandLoader
from paz.backend.boxes import to_one_hot
from processors_standard import TransposeOfArray, ListToArray

import paz.processors as pr
from paz.processors import SequentialProcessor

data_loader = RenderedHandLoader(
    '/media/jarvis/CommonFiles/5th_Semester/DFKI_Work/RHD_published_v2/')

from HandPoseEstimation import HandSegmentationNet, PosePriorNet, PoseNet
from HandPoseEstimation import ViewPointNet
import numpy as np
from pipelines import PostProcessSegmentation, \
    Process2DKeypoints
from paz.backend.image.opencv_image import load_image
from backend_keypoints import create_multiple_gaussian_map
from processors_keypoints import ExtractKeypoints

np.random.seed(0)

use_pretrained = True
HandSegNet = HandSegmentationNet()
HandPoseNet = PoseNet()
HandPosePriorNet = PosePriorNet()
HandViewPointNet = ViewPointNet()


def test_keypoints_to_palm_coordinates():
    keypoints = np.arange(0, 123).reshape((41, 3))
    keypoint_palm = keypoints_to_palm_coordinates(keypoints)
    assert keypoint_palm[LEFT_WRIST, :].all() == np.array([
        [18., 19., 20.]]).all()
    assert keypoint_palm[RIGHT_WRIST, :].all() == np.array([
        [81., 82., 83.]]).all()


def test_one_hot_encode():
    one_hot_vector = to_one_hot([1], 2)
    assert type(one_hot_vector).__module__ == np.__name__
    assert one_hot_vector.all() == np.array([0, 1]).all()
    assert to_one_hot([0], 2).all() == np.array([1, 0]).all()


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


def test_extracting_handside():
    keypoints3D = np.random.rand(42, 3)
    left_keypoints = extract_hand_side_keypoints(keypoints3D, 0)
    right_keypoints = extract_hand_side_keypoints(keypoints3D, 1)
    assert left_keypoints.shape == (21, 3)
    assert right_keypoints.shape == (21, 3)


def test_to_homogeneous():
    vector_shape = (1, 3)
    keypoint = np.zeros(vector_shape)
    homogeneous_keypoint = to_homogeneous_coordinates(keypoint)
    assert homogeneous_keypoint[-1] == 1
    assert homogeneous_keypoint.shape == (vector_shape[1] + 1,)


def test_to_translation_1D():
    translation_matrix = build_translation_matrix_SE3([1])

    assert translation_matrix.shape == (1, 4, 4)
    assert translation_matrix[-1].all() == np.array([0, 0, 0, 1]).all()


def test_to_translation_3D():
    translation_matrix = build_translation_matrix_SE3([1, 2, 3])

    assert translation_matrix[:, :, -1].all() == np.array([[1, 2, 3, 1]]).all()
    assert translation_matrix.shape == (1, 4, 4)
    assert translation_matrix[-1].all() == np.array([0, 0, 0, 1]).all()


def test_to_affine_matrix():
    matrix = np.arange(0, 9).reshape((3, 3))
    affine_matrix = build_affine_matrix(matrix)

    assert matrix.shape == (3, 3)
    assert affine_matrix.shape == (4, 4)


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


def test_rotation_matrix_axis_angles():
    rotation_matrix_test = np.array([[0.739, -0.406, 0.536],
                                     [0.536, 0.837, -0.1],
                                     [-0.4, 0.36, 0.837]])
    rotation_matrix = rotation_from_axis_angles(np.deg2rad([15, 30, 30]))
    print(rotation_matrix)
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
    transformed_keypoints, rotation_matrix = canonical_transformations_on_keypoints(
        keypoints3D.T)

    assert transformed_keypoints.shape == (42, 3)
    assert rotation_matrix.shape == (3, 3)


def test_preprocess_image():
    preprocess_pipeline = SequentialProcessor(
        [pr.NormalizeImage(), pr.ResizeImage((320, 320)), pr.ExpandDims(0)])
    image = load_image('./sample.jpg')
    processed_image = preprocess_pipeline(image)

    assert len(processed_image.shape) == 4
    assert processed_image.shape == (1, 320, 320, 3)


def test_image_cropping():
    handsegnet = HandSegmentationNet()
    preprocess_image = SequentialProcessor(
        [pr.NormalizeImage(), pr.ResizeImage((320, 320)),
         pr.ExpandDims(0)])

    postprocess_segmentation = PostProcessSegmentation(
        320, 320)

    localize_hand = pr.Predict(handsegnet, preprocess_image,
                               postprocess_segmentation)
    image = load_image('./sample.jpg')
    hand_crop, segmentation_map, center, boxes, crop_sizes = localize_hand(
        image)
    box = boxes[0]
    xmin, ymin, xmax, ymax = box
    crop_size = crop_sizes[0]

    assert len(hand_crop.shape) == 4
    assert hand_crop.shape == (1, 256, 256, 3)
    assert len(segmentation_map.shape) == 4
    assert segmentation_map.shape == (1, 320, 320, 1)
    assert center == [[191.5, 194.5]]
    assert len(box) == 4
    assert box == [114, 153, 269, 236]
    assert xmax > xmin and ymin > ymax
    assert round(crop_size[0], 2) == 1.32


def test_segmentation_postprocess():
    preprocess_pipeline = SequentialProcessor(
        [pr.NormalizeImage(), pr.ResizeImage((320, 320)), pr.ExpandDims(0)])
    image = load_image('./sample.jpg')
    processed_image = preprocess_pipeline(image)

    localization_pipeline = PostProcessSegmentation(HandSegNet)
    localization_output = localization_pipeline(processed_image)

    assert len(localization_output) == 5
    assert localization_output[0].shape == (1, 256, 256, 3)
    assert localization_output[1].shape == (1, 320, 320, 1)
    assert localization_output[2].shape == (1, 2)
    assert localization_output[3].shape == (1, 2, 2)
    assert localization_output[4].shape == (1, 1)


def test_keypoints2D_process():
    preprocess_pipeline = SequentialProcessor(
        [pr.NormalizeImage(), pr.ResizeImage((320, 320)), pr.ExpandDims(0)])
    image = load_image('./sample.jpg')
    processed_image = preprocess_pipeline(image)

    localization_pipeline = PostProcessSegmentation(HandSegNet)
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
                                                 sigma=0.1, validity_mask=None)
    gaussian_maps = np.expand_dims(gaussian_maps, axis=0)
    keypoints_extraction_pipeline = ExtractKeypoints()
    keypoints2D = keypoints_extraction_pipeline(gaussian_maps)

    assert keypoints2D[0] == [0, 0]
