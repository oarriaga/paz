import jax
import jax.numpy as jp

from paz.graphics.renderer import optics


def test_compute_new_rays_reflects_opaque_hit_above_surface():
    normal = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, -1.0]])
    point = jp.array([[0.0, 0.0, 0.0]])
    transparencies = jp.array([0.0])
    args = normal, eye, jp.array([1.0]), point, transparencies
    origin, direction = optics.compute_new_rays(*args)
    expected = jp.array([[0.0, 1.0, 1.0]]) / jp.sqrt(2.0)
    assert jp.allclose(direction, expected, atol=1e-5)
    assert jp.allclose(origin, normal * optics.BOUNCE_ORIGIN_EPSILON)


def test_compute_new_rays_refracts_head_on_ray_straight_through():
    normal = jp.array([[0.0, 0.0, -1.0]])
    eye = jp.array([[0.0, 0.0, -1.0]])
    point = jp.array([[0.0, 0.0, 0.0]])
    transparencies = jp.array([1.0])
    args = normal, eye, jp.array([1.0 / 1.5]), point, transparencies
    origin, direction = optics.compute_new_rays(*args)
    assert jp.allclose(direction, -eye, atol=1e-5)
    assert jp.allclose(origin, -normal * optics.BOUNCE_ORIGIN_EPSILON)


def test_transmission_cosine_gradient_is_zero_under_total_reflection():
    eye = jp.array([[0.8, 0.6, 0.0]])
    normal = jp.array([[0.0, 1.0, 0.0]])

    def compute_cos_transmit(n_ratio):
        cosines = optics.compute_transmission_cosines(eye, normal, n_ratio)
        return cosines[2][0]

    assert compute_cos_transmit(1.5) == 0.0
    assert jax.grad(compute_cos_transmit)(1.5) == 0.0
