import jax
import jax.numpy as jp
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
