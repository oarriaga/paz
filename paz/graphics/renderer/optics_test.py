import jax.numpy as jp

from paz.graphics.renderer import optics


def test_compute_new_rays_reflection_is_normalized():
    normal = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, -1.0]])
    point = jp.array([[0.0, 0.0, 0.0]])
    transparencies = jp.array([0.0])
    args = normal, eye, jp.array([1.0]), point, transparencies
    _, direction = optics.compute_new_rays(*args)
    norm = jp.linalg.norm(direction, axis=-1)
    assert jp.allclose(norm, 1.0, atol=1e-5)


def test_compute_new_rays_refraction_is_normalized():
    normal = jp.array([[0.0, 0.0, -1.0]])
    eye = jp.array([[0.0, 0.0, -1.0]])
    point = jp.array([[0.0, 0.0, 0.0]])
    transparencies = jp.array([1.0])
    args = normal, eye, jp.array([1.0 / 1.5]), point, transparencies
    _, direction = optics.compute_new_rays(*args)
    norm = jp.linalg.norm(direction, axis=-1)
    assert jp.allclose(norm, 1.0, atol=1e-5)
