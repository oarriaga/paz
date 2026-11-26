import jax.numpy as jp

import paz
from paz.graphics.renderer import rays
from paz.graphics.types import Material, PointLight, Scene, Sphere, Cube


def build_glass_scene(build_shape):
    glass = Material(jp.zeros(3), 0.0, 0.0, 0.0, 200.0, 0.0, 1.0, 1.5)
    return Scene([build_shape(jp.eye(4), glass)])


def trace(scene, origin, direction, num_steps):
    lights = [PointLight(jp.ones(3), jp.array([0.0, 10.0, 10.0]))]
    compiled = paz.graphics.scene.compile(scene, lights, None, None)
    state = rays.initialize_state((origin[None], direction[None]))
    for step_arg in range(num_steps):
        state = rays.bounce(state, step_arg, compiled, False, 128)
    return state.rays[1][0], state.refractive_index[0]


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


def test_glass_sphere_refracts_on_entry():
    scene = build_glass_scene(Sphere)
    origin, direction = jp.array([0.5, 0.0, 5.0]), jp.array([0.0, 0.0, -1.0])
    deviation = jp.arcsin(0.5) - jp.arcsin(0.5 / 1.5)
    expected = jp.array([-jp.sin(deviation), 0.0, -jp.cos(deviation)])
    direction, refractive_index = trace(scene, origin, direction, 1)
    assert jp.allclose(direction, expected, atol=1e-4)
    assert jp.allclose(refractive_index, 1.5)


def test_glass_sphere_refracts_on_exit():
    scene = build_glass_scene(Sphere)
    origin, direction = jp.array([0.5, 0.0, 5.0]), jp.array([0.0, 0.0, -1.0])
    deviation = 2.0 * (jp.arcsin(0.5) - jp.arcsin(0.5 / 1.5))
    expected = jp.array([-jp.sin(deviation), 0.0, -jp.cos(deviation)])
    direction, refractive_index = trace(scene, origin, direction, 2)
    assert jp.allclose(direction, expected, atol=1e-2)
    assert jp.allclose(refractive_index, 1.0)


def test_glass_cube_refracts_on_entry():
    scene = build_glass_scene(Cube)
    sine = jp.sqrt(0.5)
    origin, direction = jp.array([-2.8, 0.0, 3.0]), jp.array([sine, 0.0, -sine])
    transmitted_sine = sine / 1.5
    transmitted_cosine = jp.sqrt(1.0 - transmitted_sine**2)
    expected = jp.array([transmitted_sine, 0.0, -transmitted_cosine])
    direction, refractive_index = trace(scene, origin, direction, 1)
    assert jp.allclose(direction, expected, atol=1e-4)
    assert jp.allclose(refractive_index, 1.5)


def test_glass_cube_exit_direction_equals_entry_direction():
    scene = build_glass_scene(Cube)
    sine = jp.sqrt(0.5)
    origin, direction = jp.array([-2.8, 0.0, 3.0]), jp.array([sine, 0.0, -sine])
    exit_direction, refractive_index = trace(scene, origin, direction, 2)
    assert jp.allclose(exit_direction, direction, atol=1e-4)
    assert jp.allclose(refractive_index, 1.0)
