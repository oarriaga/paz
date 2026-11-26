import jax
import jax.numpy as jp
import pytest
from paz.models.keypoint.keypointnet import KeypointNet2D
from paz.models.keypoint.layers import ExpectedValue2D


def compute_one_hot_expected_value(H, W, row, col):
    heatmap = jp.zeros((1, 1, H, W)).at[0, 0, row, col].set(1.0)
    return ExpectedValue2D()(heatmap)[0, 0]


@pytest.mark.parametrize("H, W", [(4, 4), (4, 6), (6, 4)])
def test_expected_value_of_one_hot_is_pixel_center(H, W):
    row, col = 1, 3
    keypoint = compute_one_hot_expected_value(H, W, row, col)
    x = (col + 0.5) / (W / 2) - 1
    y = 1 - (row + 0.5) / (H / 2)
    assert jp.allclose(keypoint, jp.array([x, y]), atol=1e-6)


def test_KeypointNet2D_runs_on_non_square_input():
    model = KeypointNet2D((32, 48, 3), 5, 8)
    images = jax.random.uniform(jax.random.PRNGKey(0), (2, 32, 48, 3))
    keypoints = jax.jit(model)(images)
    assert keypoints.shape == (2, 5, 2)
    assert jp.all(jp.abs(keypoints) <= 1.0)
