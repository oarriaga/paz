import jax.numpy as jp

from paz.graphics.renderer import rays


def test_initialize_state():
    num_rays = 10
    camera_rays = jp.zeros((num_rays, 3)), jp.ones((num_rays, 3))
    state = rays.initialize_state(camera_rays)
    assert state.color.shape == (num_rays, 3)
    assert state.throughput.shape == (num_rays, 3)
    assert jp.all(state.refractive_index == 1.0)
    assert state.depth.shape == (num_rays,)
    assert state.hit_mask.dtype == bool


def test_accumulate_color():
    num_rays = 2
    colors = jp.zeros((num_rays, 3))
    throughput = jp.ones((num_rays, 3))
    active_mask = jp.array([True, False])
    intersected_colors = jp.ones((num_rays, 3))
    reflectivities = jp.zeros((num_rays,))
    transparencies = jp.zeros((num_rays,))
    args = colors, throughput, active_mask, intersected_colors
    args += reflectivities, transparencies
    result = rays.accumulate_color(*args)
    assert jp.array_equal(result[0], jp.array([1.0, 1.0, 1.0]))
    assert jp.array_equal(result[1], jp.array([0.0, 0.0, 0.0]))
