import pytest
import jax
import jax.numpy as jp
import paz
from paz import SE3
from paz.graphics import renderer
from paz.graphics.types import (
    Shape,
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
from paz.graphics import constants

# --- Helper Functions Unit Tests ---


def test_take_closest():
    array = jp.array(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]
    )
    indices = jp.array([0, 1, 0])
    expected = jp.array([[1, 1, 1], [5, 5, 5], [3, 3, 3]])
    result = renderer.take_closest(array, indices)
    assert jp.array_equal(result, expected)


# def test_compute_soft_occlusion():
#     slope = 10.0
#     norms = jp.array([10.0, 10.0, 10.0])
#     depths = jp.array([10.0, 5.0, 10.0])
#     hit_mask = jp.array([False, True, True])
#     res = renderer.compute_soft_occlusion(norms, depths, hit_mask, slope=slope)
#     assert res[0] == 0.0
#     assert res[1] > 0.9
#     assert res[2] == 0.5


def test_compute_scene_hit_mask():
    hit_masks = jp.array([[False, True, False], [False, False, True]])
    expected = jp.array([False, True, True])
    result = renderer.compute_scene_hit_mask(hit_masks)
    assert jp.array_equal(result, expected)


# def test_compute_occlusion_binary():
#     norms = jp.array([10.0, 10.0])
#     depths = jp.array([5.0, 10.0])
#     mask = jp.array([True, True])
#     res = renderer.compute_occlusion(norms, depths, mask)
#     assert jp.array_equal(res, jp.array([True, False]))


def test_select_colors():
    depths = jp.array([[10.0, 2.0], [5.0, 8.0]])
    colors = jp.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]])
    expected = jp.array([[0, 0, 1], [0, 1, 0]])
    result = renderer.select_colors(depths, colors)
    assert jp.array_equal(result, expected)


# --- Internal Helper Function Tests ---


def test_initialize_render_state():
    num_rays = 10
    rays = (jp.zeros((num_rays, 3)), jp.ones((num_rays, 3)))
    state = renderer.initialize_state(rays)
    assert state["color"].shape == (num_rays, 3)
    assert state["throughput"].shape == (num_rays, 3)
    assert jp.all(state["current_refractive_index"] == 1.0)
    assert state["depth"].shape == (num_rays,)
    assert state["hit_mask"].dtype == bool


def test_find_closest_intersection_args():
    hit_masks = jp.array([[True, False], [False, True]])
    depths = jp.array([[1.0, 10.0], [10.0, 2.0]])
    indices = renderer.find_closest_intersection_args(hit_masks, depths)
    assert jp.array_equal(indices, jp.array([0, 1]))


def test_get_material_properties():
    mat1 = Material(reflective=0.5)
    mat2 = Material(transparency=0.8)
    shape1 = Sphere(material=mat1)
    shape2 = Sphere(material=mat2)
    shapes = [shape1, shape2]
    indices = jp.array([0, 1])
    props = renderer.get_material_properties(shapes, indices)
    assert props["reflective"][0] == 0.5
    assert props["transparency"][1] == 0.8


def test_accumulate_local_color():
    num_rays = 2
    state = {
        "color": jp.zeros((num_rays, 3)),
        "throughput": jp.ones((num_rays, 3)),
        "active_mask": jp.array([True, False]),
    }
    local_color = jp.ones((num_rays, 3))
    materials = {
        "reflective": jp.array([0.0, 0.0]),
        "transparency": jp.array([0.0, 0.0]),
    }

    # Corrected call: _accumulate_local_color updates state in place, returns None
    renderer.accumulate_local_color(state, local_color, materials)

    # Ray 0: active, color added (1.0). Ray 1: inactive, no change (0.0).
    assert jp.array_equal(state["color"][0], jp.array([1.0, 1.0, 1.0]))
    assert jp.array_equal(state["color"][1], jp.array([0.0, 0.0, 0.0]))


# def test_compute_light_vectors():
#     light = PointLight(
#         jp.array([1.0, 1.0, 1.0]), position=jp.array([10.0, 0.0, 0.0])
#     )
#     point = jp.array([[0.0, 0.0, 0.0]])
#     direction, distance = renderer.compute_light_vectors(light, point)
#     assert jp.allclose(direction, jp.array([[1.0, 0.0, 0.0]]))
#     assert jp.allclose(distance, jp.array([10.0]))


# def test_resolve_shadow_masks():
#     mask = jp.array([True, False])  # Object 0 casts shadow, 1 does not
#     shadow_init = None
#     hits = (None, None, None, None, None, jp.array([0, 1]))  # Hit indices
#     shape_transparencies = jp.zeros(2)

#     sh_masks = renderer.resolve_shadow_masks(
#         mask,
#         shadow_init,
#         (jp.ones((2, 1), dtype=bool), *hits[1:]),
#         shape_transparencies,
#     )
#     # Object 0 -> True & True = True. Object 1 -> True & False = False.
#     assert sh_masks[0, 0] == True
#     assert sh_masks[1, 0] == False


# --- Integration Tests with Scenes ---


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


@pytest.fixture
def rays(small_image_shape, camera_pose):
    return paz.graphics.camera.build_rays(
        small_image_shape, jp.pi / 3.0, camera_pose
    )


def test_render_reflection_scene(small_image_shape, camera_pose, rays):
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
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 10.0, 5.0]))]
    scene = Scene([floor, sphere, cube])
    image, depth = renderer.render(
        small_image_shape,
        camera_pose,
        rays,
        scene,
        lights,
        mask=None,
        shadows=False,
    )
    assert image.shape == (small_image_shape[0], small_image_shape[1], 3)
    assert not jp.isnan(image).any()
    assert jp.std(image) > 0.0


def test_render_shadows_logic(small_image_shape, camera_pose, rays):
    camera_pose_shadow = SE3.view_transform(
        jp.array([0.0, 0.0, 10.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    rays_shadow = paz.graphics.camera.build_rays(
        small_image_shape, jp.pi / 3.0, camera_pose_shadow
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
    img_shadows_on, _ = renderer.render(
        small_image_shape,
        camera_pose_shadow,
        rays_shadow,
        scene_blocked,
        lights,
        mask=None,
        shadows=True,
    )
    img_shadows_off, _ = renderer.render(
        small_image_shape,
        camera_pose_shadow,
        rays_shadow,
        scene_blocked,
        lights,
        mask=None,
        shadows=False,
    )
    assert not jp.array_equal(img_shadows_on, img_shadows_off)


def test_render_shadow_mask(small_image_shape, camera_pose, rays):
    camera_pose_shadow = SE3.view_transform(
        jp.array([0.0, 0.0, 10.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    rays_shadow = paz.graphics.camera.build_rays(
        small_image_shape, jp.pi / 3.0, camera_pose_shadow
    )
    wall = Plane(
        SE3.rotation_x(jp.pi / 2), Material(color=jp.array([1.0, 1.0, 1.0]))
    )
    blocker = Sphere(SE3.translation(jp.array([0.0, 0.0, 2.0])))
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([5.0, 5.0, 5.0]))]
    scene = Scene([wall, blocker])
    shadow_mask = jp.array([True, False])
    img_no_cast, _ = renderer.render(
        small_image_shape,
        camera_pose_shadow,
        rays_shadow,
        scene,
        lights,
        mask=None,
        shadows=True,
        shadow_mask=shadow_mask,
    )
    img_cast, _ = renderer.render(
        small_image_shape,
        camera_pose_shadow,
        rays_shadow,
        scene,
        lights,
        mask=None,
        shadows=True,
        shadow_mask=None,
    )
    assert not jp.array_equal(img_no_cast, img_cast)


def test_render_masked_objects(small_image_shape, camera_pose, rays):
    sphere = Sphere(SE3.translation(jp.array([0.0, 0.0, 0.0])))
    scene = Scene([sphere])
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 0.0, 5.0]))]
    mask = jp.array([False])
    img_hidden, _ = renderer.render(
        small_image_shape,
        camera_pose,
        rays,
        scene,
        lights,
        mask=mask,
        shadows=False,
    )
    assert jp.all(img_hidden == 1.0)
    mask = jp.array([True])
    img_visible, _ = renderer.render(
        small_image_shape,
        camera_pose,
        rays,
        scene,
        lights,
        mask=mask,
        shadows=False,
    )
    assert not jp.all(img_visible == 1.0)


def test_postprocess_outputs():
    pass


def test_max_bounces_effect(small_image_shape, camera_pose, rays):
    # Use +Z camera setup which is known to work
    camera_pose_back = SE3.view_transform(
        jp.array([0.0, 0.0, 5.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    rays_back = paz.graphics.camera.build_rays(
        small_image_shape, jp.pi / 3.0, camera_pose_back
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
    shapes, mask_c, _, lights_c = paz.graphics.scene.compile(scene, lights, None)
    img_1b, _ = renderer.render_bounced(
        *small_image_shape,
        camera_pose_back,
        rays_back,
        shapes,
        lights_c,
        mask_c,
        False,
        None,
        num_bounces=1,
    )
    img_2b, _ = renderer.render_bounced(
        *small_image_shape,
        camera_pose_back,
        rays_back,
        shapes,
        lights_c,
        mask_c,
        False,
        None,
        num_bounces=2,
    )
    assert not jp.array_equal(img_1b, img_2b)
    assert not jp.all(img_1b == 1.0)


def test_jit_compilation_full(small_image_shape, camera_pose, rays):
    material = Material(color=jp.array([1.0, 0.0, 0.0]))
    sphere = Sphere(jp.eye(4), material)
    scene = Scene([sphere])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 10.0, 0.0]))]

    @jax.jit
    def jitted_render():
        return renderer.render(
            small_image_shape,
            camera_pose,
            rays,
            scene,
            lights,
            mask=None,
            shadows=True,
        )

    image, depth = jitted_render()
    assert image.shape == (small_image_shape[0], small_image_shape[1], 3)
