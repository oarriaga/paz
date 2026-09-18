import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

from paz.backend import pinhole


def test_build_cube_points3D_shape_and_center():
    cube = np.asarray(pinhole.build_cube_points3D(2.0, 4.0, 6.0))
    assert cube.shape == (8, 3)
    assert np.allclose(cube.mean(axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(np.abs(cube).max(axis=0), [1.0, 2.0, 3.0])
