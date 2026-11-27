import jax
import jax.numpy as jp
import numpy as np
import paz


def test_resize_intrinsics_keeps_rays():
    rows = [[611.1, 0.0, 315.5], [0.0, 610.0, 237.7], [0.0, 0.0, 1.0]]
    intrinsics = jp.array(rows)
    resized = paz.pinhole.resize_intrinsics(intrinsics, 0.5)
    pixels = jp.array([[0.0, 0.0], [10.0, 7.0], [319.0, 239.0]])
    full_pixels = (pixels + 0.5) * 2.0 - 0.5
    with jax.default_matmul_precision("highest"):
        args = resized, pixels, jp.eye(4)
        _, small = paz.graphics.camera.build_pixel_rays(*args)
        args = intrinsics, full_pixels, jp.eye(4)
        _, full = paz.graphics.camera.build_pixel_rays(*args)
    assert jp.allclose(small, full, atol=1e-5)


def make_chessboard(num_rows=7, num_columns=10, square_size=40):
    board = np.full((num_rows + 2, num_columns + 2), 255, np.uint8)
    board[1:-1:2, 1:-1:2] = 0
    board[2:-1:2, 2:-1:2] = 0
    return np.kron(board, np.ones((square_size, square_size), np.uint8))


def test_calibrate_accepts_RGB_images():
    gray = make_chessboard()
    RGB = np.repeat(gray[..., None], 3, axis=-1)
    gray_matrix, gray_distortion = paz.pinhole.calibrate([gray, gray], (6, 9))
    RGB_matrix, RGB_distortion = paz.pinhole.calibrate([RGB, RGB], (6, 9))
    assert RGB_matrix.shape == (3, 3)
    assert np.allclose(RGB_matrix, gray_matrix)
    assert np.allclose(RGB_distortion, gray_distortion)
