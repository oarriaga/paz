import numpy as np
import backend as B
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


@pytest.mark.parametrize("heatmap_sum_shape", [(1, 17,  256, 352)])
def test_get_heatmap_sum(output, num_joints, heatmap_sum_shape):
    heatmap_sum = B.get_heatmap_sum(output, num_joints, 0)

    assert (heatmap_sum.shape == heatmap_sum_shape)


@pytest.mark.parametrize("heatmap_sum_shape", [(1, 17,  256, 352)])
def test_get_heatmap_sum_with_flip(
        output, num_joints, indices, heatmap_sum_shape):
    heatmap_sum = B.get_heatmap_sum_with_flip(output, num_joints, indices, 0)

    assert (heatmap_sum.shape == heatmap_sum_shape)


@pytest.mark.parametrize("tags_shape", [(17,  256, 352)])
def test_get_tags(output, num_joints, tags_shape):
    tags = B.get_tags(output, num_joints)
    print(tags[0].shape)
    assert (tags[0].shape == tags_shape)


@pytest.mark.parametrize("tags_shape", [(17,  256, 352)])
def test_get_tags_with_flip(output, num_joints, indices, tags_shape):
    tags = B.get_tags_with_flip(output, num_joints, indices)
    assert (tags[0].shape == tags_shape)
