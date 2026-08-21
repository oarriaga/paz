from pathlib import Path

import pytest
import jax
import jax.numpy as jp
import paz
from paz import SE3
from paz.graphics import renderer
from paz.graphics.types import (
    Material,
    PointLight,
    Sphere,
    Cube,
    Plane,
    Cylinder,
    Cone,
    SphericalPattern,
    PlanarPattern,
    CylindricalPattern,
    Scene,
)


@pytest.fixture
def small_image_shape():
    return 32, 32


@pytest.fixture
def camera_pose():
    return SE3.view_transform(
        jp.array([0.0, 0.0, 5.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )


def snapshot_path(filename):
    return str(Path(__file__).parent / "snapshots" / filename)


def assert_snapshot(array, filename, atol=1e-4):
    paz.assert_snapshot(array, snapshot_path(filename), atol=atol)


def render_scene(image_shape, camera_pose, scene, lights, mask=None,
                 shadows=False, tiles=(1, 1), chunk_size=1024,
                 shadow_mask=None, num_bounces=1):
    args = image_shape, jp.pi / 3.0, camera_pose, scene, mask, lights
    args += tiles, chunk_size
    return renderer.render(*args, shadows, shadow_mask, num_bounces)


def compute_max_abs_difference(array_A, array_B):
    return float(jp.max(jp.abs(array_A - array_B)))


def assert_render_matches(actual, expected, atol=1e-4):
    actual_image, actual_depth = actual
    expected_image, expected_depth = expected
    assert compute_max_abs_difference(actual_image, expected_image) <= atol
    assert compute_max_abs_difference(actual_depth, expected_depth) <= atol


def build_shadow_scene():
    material = Material(
        color=jp.array([1.0, 0.0, 0.0]),
        ambient=0.1,
        diffuse=0.9,
        specular=0.3,
        shininess=200.0,
    )
    sphere = Sphere(SE3.translation(jp.array([0.0, 1.0, 0.0])), material)
    plane = Plane()
    scene = Scene([sphere, plane])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 3.0, -3.0]))]
    return scene, lights


def build_shaded_sphere_scene():
    material = Material(
        color=jp.array([0.8, 0.2, 0.1]),
        ambient=0.1,
        diffuse=0.9,
        specular=0.3,
        shininess=200.0,
    )
    lights = [PointLight(jp.ones(3), jp.array([0.0, 5.0, -5.0]))]
    return Scene([Sphere(jp.eye(4), material)]), lights


def build_material_scene():
    mirror_material = Material(color=jp.array([1.0, 1.0, 1.0]), reflective=0.8)
    glass_material = Material(
        color=jp.array([0.9, 0.9, 1.0]),
        transparency=0.9,
        refractive_index=1.5,
    )
    floor_material = Material(color=jp.array([0.5, 0.5, 0.5]))
    floor = Plane(jp.eye(4), floor_material)
    sphere = Sphere(
        SE3.translation(jp.array([-1.5, 0.0, 0.0])), mirror_material
    )
    cube = Cube(SE3.translation(jp.array([1.5, 0.0, 0.0])), glass_material)
    lights = [PointLight(jp.ones(3), jp.array([0.0, 10.0, 5.0]))]
    return Scene([floor, sphere, cube]), lights


def build_checkered_image(box_size=4, rows=4, cols=4):
    green = jp.array([85 / 255, 181 / 255, 103 / 255])
    white = jp.ones(3)
    checkered = jp.indices((rows, cols)).sum(axis=0) % 2
    channels = []
    for channel in range(3):
        values = jp.kron(checkered, jp.ones((box_size, box_size)))
        values = green[channel] * values + white[channel] * (1 - values)
        channels.append(jp.expand_dims(values, axis=-1))
    return jp.concatenate(channels, axis=-1)


def build_primitives_snapshot_scene():
    pattern = build_checkered_image()
    material = Material(jp.zeros(3), 0.3, 0.1, 0.0, 100.0)
    sphere = Sphere(
        SE3.translation(jp.array([0.0, 1.0, -2.0])),
        material,
        SphericalPattern(pattern),
    )
    cylinder = Cylinder(
        SE3.translation(jp.array([-1.2, 0.7, 0.0]))
        @ SE3.scaling(jp.full(3, 0.7)),
        material,
        CylindricalPattern(pattern),
    )
    cone = Cone(
        SE3.translation(jp.array([1.2, 0.7, 0.0]))
        @ SE3.scaling(jp.full(3, 0.7)),
        material,
        PlanarPattern(pattern),
    )
    return Scene([sphere, cylinder, cone, Plane()])


def build_shadow_snapshot_scene():
    wall = Plane(
        SE3.rotation_x(jp.pi / 2),
        Material(color=jp.array([1.0, 1.0, 1.0])),
    )
    blocker = Sphere(
        SE3.translation(jp.array([0.0, 0.0, 2.0]))
        @ SE3.scaling(jp.full(3, 0.5))
    )
    return Scene([wall, blocker])


def build_bounce_snapshot_scene():
    mirror = Material(
        color=jp.array([0.0, 0.0, 0.0]),
        reflective=1.0,
        diffuse=0.0,
        ambient=0.0,
    )
    red = Material(color=jp.array([1.0, 0.0, 0.0]), ambient=1.0)
    sphere = Sphere(SE3.scaling(jp.full(3, 2.0)), mirror)
    target = Sphere(
        SE3.translation(jp.array([0.0, 0.0, 10.0]))
        @ SE3.scaling(jp.full(3, 2.0)),
        red,
    )
    return Scene([sphere, target])


def build_shifted_sphere_scene(z_shift):
    offset = jp.array([0.0, 0.0, 1.0]) * z_shift
    material = Material(color=jp.array([0.8, 0.2, 0.1]))
    return Scene([Sphere(SE3.translation(offset), material)])


def test_render_reflection_scene(small_image_shape, camera_pose):
    scene, lights = build_material_scene()
    image, depth = render_scene(small_image_shape, camera_pose, scene, lights)
    assert image.shape == (small_image_shape[0], small_image_shape[1], 3)
    assert not jp.isnan(image).any()
    assert jp.std(image) > 0.0


def test_render_primitives_matches_snapshot(camera_pose):
    scene = build_primitives_snapshot_scene()
    lights = [PointLight(jp.ones(3), jp.array([0.0, 5.0, -5.0]))]
    image, depth = render_scene((24, 24), camera_pose, scene, lights)
    assert_snapshot(image, "renderer_primitives_image.npy", atol=1e-3)
    assert_snapshot(depth, "renderer_primitives_depth.npy", atol=3e-3)


def test_render_shadow_mask_matches_snapshot():
    camera_pose = SE3.view_transform(
        jp.array([0.0, 0.0, 10.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    scene = build_shadow_snapshot_scene()
    lights = [PointLight(jp.ones(3), jp.array([0.0, 0.0, 5.0]))]
    shadow_mask = jp.array([True, False])
    image, depth = render_scene(
        (24, 24),
        camera_pose,
        scene,
        lights,
        shadows=True,
        shadow_mask=shadow_mask,
    )
    assert_snapshot(image, "renderer_shadow_image.npy", atol=1e-3)
    assert_snapshot(depth, "renderer_shadow_depth.npy", atol=3e-3)


def test_render_bounces_match_snapshot(camera_pose):
    scene = build_bounce_snapshot_scene()
    lights = [PointLight(jp.ones(3), jp.array([0.0, 0.0, 5.0]))]
    image, depth = render_scene(
        (24, 24),
        camera_pose,
        scene,
        lights,
        num_bounces=2,
    )
    assert_snapshot(image, "renderer_bounce_image.npy", atol=1e-3)
    assert_snapshot(depth, "renderer_bounce_depth.npy", atol=3e-3)


def test_render_gradient_matches_snapshot(camera_pose):
    lights = [PointLight(jp.ones(3), jp.array([0.0, 5.0, -5.0]))]

    def loss(shift):
        scene = build_shifted_sphere_scene(shift[0])
        image, depth = render_scene((24, 24), camera_pose, scene, lights)
        return jp.mean(depth) + 0.01 * jp.mean(image)

    gradient = jax.grad(loss)(jp.array([0.1]))
    assert_snapshot(gradient, "renderer_shift_gradient.npy", atol=2e-3)


def test_render_rect_tiles_match_single_tile(small_image_shape, camera_pose):
    scene, lights = build_shaded_sphere_scene()
    expected = render_scene(small_image_shape, camera_pose, scene, lights)
    actual = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        tiles=(2, 4),
        chunk_size=13,
    )
    assert_render_matches(actual, expected, atol=5e-4)


def test_render_rect_tiles_match_shadows(small_image_shape, camera_pose):
    scene, lights = build_shadow_scene()
    expected = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        shadows=True,
    )
    actual = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        shadows=True,
        tiles=(2, 2),
        chunk_size=17,
    )
    assert_render_matches(actual, expected)


def test_render_depth_is_chunk_invariant(small_image_shape, camera_pose):
    scene, lights = build_shaded_sphere_scene()
    _, expected_depth = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        chunk_size=1024,
    )
    _, actual_depth = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        chunk_size=11,
    )
    assert compute_max_abs_difference(actual_depth, expected_depth) <= 1e-4


def test_render_gradient_is_chunk_invariant(small_image_shape, camera_pose):
    lights = [PointLight(jp.ones(3), jp.array([0.0, 5.0, -5.0]))]

    def large_chunk_loss(shift):
        scene = build_shifted_sphere_scene(shift[0])
        _, depth = render_scene(small_image_shape, camera_pose, scene, lights)
        return jp.mean(depth)

    def small_chunk_loss(shift):
        scene = build_shifted_sphere_scene(shift[0])
        _, depth = render_scene(
            small_image_shape,
            camera_pose,
            scene,
            lights,
            chunk_size=11,
        )
        return jp.mean(depth)

    shift = jp.array([0.1])
    large_gradient = jax.grad(large_chunk_loss)(shift)
    small_gradient = jax.grad(small_chunk_loss)(shift)
    assert jp.abs(large_gradient[0]) > 1e-5
    assert jp.allclose(small_gradient, large_gradient, atol=5e-4)


def test_render_jit_compatible(small_image_shape, camera_pose):
    scene, lights = build_shaded_sphere_scene()

    @jax.jit
    def jitted_render():
        return render_scene(
            small_image_shape,
            camera_pose,
            scene,
            lights,
            tiles=(2, 2),
            chunk_size=16,
        )

    image, depth = jitted_render()
    assert image.shape == (small_image_shape[0], small_image_shape[1], 3)
    assert depth.shape == small_image_shape


def test_render_shadows_logic(small_image_shape, camera_pose):
    camera_pose_shadow = SE3.view_transform(
        jp.array([0.0, 0.0, 10.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    wall = Plane(
        SE3.rotation_x(jp.pi / 2), Material(color=jp.array([1.0, 1.0, 1.0]))
    )
    blocker = Sphere(
        SE3.translation(jp.array([0.0, 0.0, 2.0]))
        @ SE3.scaling(jp.full(3, 0.5))
    )
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 0.0, 5.0]))]
    scene_blocked = Scene([wall, blocker])
    img_shadows_on, _ = render_scene(
        small_image_shape,
        camera_pose_shadow,
        scene_blocked,
        lights,
        shadows=True,
    )
    img_shadows_off, _ = render_scene(
        small_image_shape,
        camera_pose_shadow,
        scene_blocked,
        lights,
        shadows=False,
    )
    assert not jp.array_equal(img_shadows_on, img_shadows_off)


def test_render_single_sphere_shadows_stay_local():
    image_shape = (60, 80)
    material = Material(
        color=jp.array([1.0, 0.0, 0.0]),
        ambient=0.1,
        diffuse=0.9,
        specular=0.3,
        shininess=200.0,
    )
    sphere = Sphere(SE3.translation(jp.array([0.0, 1.0, 0.0])), material)
    scene = Scene([sphere])
    camera_pose = SE3.view_transform(
        jp.array([3.0, 3.0, 0.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    lights = [PointLight(jp.ones(3), jp.array([0.0, 3.0, -3.0]))]
    image_no_shadows, depth_no_shadows = render_scene(
        image_shape, camera_pose, scene, lights
    )
    image_shadows, depth_shadows = render_scene(
        image_shape, camera_pose, scene, lights, shadows=True
    )
    sphere_mask = depth_no_shadows > 0.0
    diff_mask = jp.any(
        jp.abs(image_no_shadows - image_shadows) > 1e-4, axis=-1
    )
    background_mask = ~sphere_mask
    assert jp.allclose(depth_no_shadows, depth_shadows)
    assert not jp.isnan(image_shadows).any()
    assert diff_mask.any()
    assert jp.all(jp.logical_or(~diff_mask, sphere_mask))
    assert jp.allclose(
        image_no_shadows[background_mask], image_shadows[background_mask]
    )


def test_render_shadow_mask(small_image_shape, camera_pose):
    camera_pose_shadow = SE3.view_transform(
        jp.array([0.0, 0.0, 10.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    wall = Plane(
        SE3.rotation_x(jp.pi / 2), Material(color=jp.array([1.0, 1.0, 1.0]))
    )
    blocker = Sphere(SE3.translation(jp.array([0.0, 0.0, 2.0])))
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([5.0, 5.0, 5.0]))]
    scene = Scene([wall, blocker])
    shadow_mask = jp.array([True, False])
    img_no_cast, _ = render_scene(
        small_image_shape,
        camera_pose_shadow,
        scene,
        lights,
        shadows=True,
        shadow_mask=shadow_mask,
    )
    img_cast, _ = render_scene(
        small_image_shape,
        camera_pose_shadow,
        scene,
        lights,
        shadows=True,
        shadow_mask=None,
    )
    assert not jp.array_equal(img_no_cast, img_cast)


def test_render_masked_objects(small_image_shape, camera_pose):
    sphere = Sphere(SE3.translation(jp.array([0.0, 0.0, 0.0])))
    scene = Scene([sphere])
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 0.0, 5.0]))]
    mask = jp.array([False])
    img_hidden, _ = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        mask=mask,
    )
    assert jp.all(img_hidden == 1.0)
    mask = jp.array([True])
    img_visible, _ = render_scene(
        small_image_shape,
        camera_pose,
        scene,
        lights,
        mask=mask,
    )
    assert not jp.all(img_visible == 1.0)


def test_render_masks_returns_shape_masks(small_image_shape, camera_pose):
    scene, lights = build_material_scene()
    depth = 0.1, 10.0
    args = small_image_shape, jp.pi / 3.0, camera_pose, scene, lights
    masks = renderer.render_masks(*args, depth, (2, 2), 16, num_objects=2)
    assert masks.shape == (2, small_image_shape[0], small_image_shape[1], 1)
    assert jp.any(masks > 0.0)


def test_max_bounces_effect(small_image_shape, camera_pose):
    camera_pose_back = SE3.view_transform(
        jp.array([0.0, 0.0, 5.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    mirror_mat = Material(
        color=jp.array([0.0, 0.0, 0.0]),
        reflective=1.0,
        diffuse=0.0,
        ambient=0.0,
    )
    red_mat = Material(color=jp.array([1.0, 0.0, 0.0]), ambient=1.0)
    mirror = Sphere(SE3.scaling(jp.full(3, 2.0)), mirror_mat)
    red_obj = Sphere(
        SE3.translation(jp.array([0.0, 0.0, 10.0]))
        @ SE3.scaling(jp.full(3, 2.0)),
        red_mat,
    )
    scene = Scene([mirror, red_obj])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 0.0, 5.0]))]
    img_1b, _ = render_scene(
        small_image_shape,
        camera_pose_back,
        scene,
        lights,
        num_bounces=1,
    )
    img_2b, _ = render_scene(
        small_image_shape,
        camera_pose_back,
        scene,
        lights,
        num_bounces=2,
    )
    assert not jp.array_equal(img_1b, img_2b)
    assert not jp.all(img_1b == 1.0)


def test_scene_renderer_returns_uint8_frame(small_image_shape, camera_pose):
    material = Material(color=jp.array([1.0, 0.0, 0.0]))
    sphere = Sphere(jp.eye(4), material)
    scene = Scene([sphere])
    render_frame = paz.graphics.scene_renderer(
        scene,
        small_image_shape[0],
        small_image_shape[1],
        jp.pi / 3.0,
        shadows=True,
    )
    image = render_frame(camera_pose)
    assert image.shape == (small_image_shape[0], small_image_shape[1], 3)
    assert image.dtype == jp.uint8


def test_jit_compilation_full(small_image_shape, camera_pose):
    material = Material(color=jp.array([1.0, 0.0, 0.0]))
    sphere = Sphere(jp.eye(4), material)
    scene = Scene([sphere])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 10.0, 0.0]))]

    @jax.jit
    def jitted_render():
        return render_scene(small_image_shape, camera_pose, scene, lights)

    image, depth = jitted_render()
    assert image.shape == (small_image_shape[0], small_image_shape[1], 3)
