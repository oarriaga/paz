import numpy as np
from paz.backend import heatmaps
import pytest


@pytest.fixture(params=[17])
def num_joints(request):
    return (request.param)


@pytest.fixture(params=[np.ones((1, 34,  256, 352))])
def output(request):
    return (request.param)


@pytest.fixture(params=[[0, 2, 1, 4, 3, 6, 5, 8, 7, 10,
                         9, 12, 11, 14, 13, 16, 15]])
def indices(request):
    return request.param


@pytest.fixture()
def keypoint_indices():
    return np.array([[[130207, 103661, 222371, 297618, 214202, 215203,
                       205997, 161834, 158249, 205476, 160803, 214189,
                       157218, 153142, 226989, 223418, 222381,  38569,
                       225462,  75892, 143125, 159411,  91339,  83051,
                       160870, 159334, 161859, 147715, 160862, 158815]]])


@pytest.fixture()
def keypoints_locations():
    return np.array([[319, 369], [173, 294], [259, 631],
                     [178, 845], [186, 608], [131, 611],
                     [77, 585], [266, 459], [201, 449],
                     [260, 583], [291, 456], [173, 608],
                     [226, 446], [22, 435], [301, 644],
                     [250, 634], [269, 631], [201, 109],
                     [182, 640], [212, 215], [213, 406],
                     [307, 452], [171, 259], [331, 235],
                     [6, 457], [230, 452], [291, 459],
                     [227, 419], [350, 456], [63, 451]])


@pytest.fixture()
def detections():
    return np.array([[1.59, 2.54, 1.71, -1.0, 1.0],
                     [2.37, 2.02, 1.70, 1.00, 0.0],
                     [1.63, 4.34, 2.52e-03, 0.00, 0.0],
                     [1.46, 5.81, 2.50e-03, 0.00, 1.0],
                     [1.86, 4.18, 2.36e-03, 0.00, 0.0]])


@pytest.fixture()
def valid_detections():
    return np.array([[1.59, 2.54, 1.71, -1.0, 1.0],
                     [2.37, 2.02, 1.70, 1.00, 0.0]])


@pytest.mark.parametrize("keypoints_heatmap_shape", [(1, 17,  256, 352)])
def test_get_keypoints_heatmap(output, num_joints, indices,
                               keypoints_heatmap_shape):
    keypoints = heatmaps.get_keypoints_heatmap(output, num_joints)
    keypoints_with_indices = heatmaps.get_keypoints_heatmap(output, num_joints,
                                                            indices)
    assert (keypoints.shape == keypoints_heatmap_shape)
    assert (keypoints_with_indices.shape == keypoints_heatmap_shape)


@pytest.mark.parametrize("tags_heatmap_shape", [(17,  256, 352)])
def test_get_tags_heatmap(output, num_joints, indices, tags_heatmap_shape):
    tags = heatmaps.get_tags_heatmap(output, num_joints)
    tags_with_indices = heatmaps.get_tags_heatmap(output, num_joints, indices)
    assert (tags[0].shape == tags_heatmap_shape)
    assert (tags_with_indices[0].shape == tags_heatmap_shape)


def test_get_keypoints_coordinates(keypoint_indices, output,
                                   keypoints_locations):
    width = output.shape[-1]
    locations = heatmaps.get_keypoints_locations(keypoint_indices, width)
    assert np.allclose(locations, keypoints_locations)


def test_get_valid_detections(detections, valid_detections):
    estimated_detection = heatmaps.get_valid_detections(detections, 0.2)
    assert np.allclose(estimated_detection, valid_detections)
