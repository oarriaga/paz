import numpy as np
import jax.numpy as jp
import paz
from paz.backend import draw


def make_camera_matrix():
    return jp.array(
        [
            [32.0, 0.0, 32.0, 0.0],
            [0.0, 32.0, 32.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


def make_transform():
    return jp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_poses_preserves_shape_and_dtype():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    transforms = jp.stack([make_transform()])
    camera_matrix = make_camera_matrix()
    result = paz.draw.poses(image, transforms, camera_matrix)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(result != 0)


def test_poses_accepts_explicit_colors():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    transforms = jp.stack([make_transform()])
    camera_matrix = make_camera_matrix()
    result = paz.draw.poses(
        image,
        transforms,
        camera_matrix,
        colors=((10, 20, 30),),
    )
    expected = np.array([10, 20, 30], dtype=np.uint8)
    assert np.any(np.all(result == expected, axis=-1))


def test_lincolors_uses_cache():
    before = draw._lincolors.cache_info().hits
    draw._lincolors(3)
    draw._lincolors(3)
    after = draw._lincolors.cache_info().hits
    assert after == before + 1
