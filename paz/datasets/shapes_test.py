import numpy as np
import pytest
import paz


@pytest.mark.parametrize("H, W", [(128, 128), (128, 160), (160, 128)])
def test_shape_boxes_lie_inside_the_image(H, W):
    for seed in range(20):
        np.random.seed(seed)
        _, _, boxes, _ = paz.datasets.load("SHAPES", H, W, 0.3, 3, 4)
        boxes = np.concatenate(boxes)
        assert np.all(boxes[:, :2] >= 0)
        assert np.all(boxes[:, 2] < W) and np.all(boxes[:, 3] < H)
