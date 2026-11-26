import jax
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


def test_transmission_cosine_gradient_is_zero_under_total_reflection():
    eye = jp.array([[0.8, 0.6, 0.0]])
    normal = jp.array([[0.0, 1.0, 0.0]])

    def compute_cos_transmit(n_ratio):
        cosines = optics.compute_transmission_cosines(eye, normal, n_ratio)
        return cosines[2][0]

    assert compute_cos_transmit(1.5) == 0.0
    assert jax.grad(compute_cos_transmit)(1.5) == 0.0
