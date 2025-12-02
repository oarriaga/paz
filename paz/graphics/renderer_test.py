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
    # shape: (T, N, 3) -> (2, 3, 3)
    # indices: (N,) -> (3,)
    array = jp.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],  # T=0
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],  # T=1
        ]
    )
    indices = jp.array([0, 1, 0])  # Select T=0 for idx 0, T=1 for idx 1, T=0 for idx 2
    expected = jp.array([[1, 1, 1], [5, 5, 5], [3, 3, 3]])
    result = renderer.take_closest(array, indices)
    assert jp.array_equal(result, expected)


def test_compute_soft_occlusion():
    slope = 10.0
    norms = jp.array([10.0, 10.0, 10.0])
    depths = jp.array([10.0, 5.0, 10.0])
    hit_mask = jp.array([False, True, True])

    res = renderer.compute_soft_occlusion(norms, depths, hit_mask, slope=slope)

    assert res[0] == 0.0  # Mask false -> 0 occlusion
    # depth < norm -> in shadow.
    # occlusion_value = 5 - 10 = -5.
    # sigmoid(-10 * -5) = sigmoid(50) ~ 1.0. Fully occluded.
    assert res[1] > 0.9
    # depth == norm -> boundary.
    # occlusion_value = 0.
    # sigmoid(0) = 0.5.
    assert res[2] == 0.5


def test_compute_scene_hit_mask():
    hit_masks = jp.array([[False, True, False], [False, False, True]])
    expected = jp.array([False, True, True])
    result = renderer.compute_scene_hit_mask(hit_masks)
    assert jp.array_equal(result, expected)


def test_compute_occlusion_binary():
    # Test binary logic helper if used (currently soft used mostly, but function exists)
    norms = jp.array([10.0, 10.0])
    depths = jp.array([5.0, 10.0])
    mask = jp.array([True, True])
    # norms > depths -> shadow
    # 10 > 5 -> True
    # 10 > 10 -> False
    res = renderer.compute_occlusion(norms, depths, mask)
    assert jp.array_equal(res, jp.array([True, False]))


def test_select_colors():
    # depths: (T, N)
    # colors: (T, N, 3)
    depths = jp.array([[10.0, 2.0], [5.0, 8.0]])  # T=0, T=1
    colors = jp.array(
        [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]]  # Red, Green  # Blue, Yellow
    )
    # Ray 0: min depth is 5.0 at T=1 -> Blue
    # Ray 1: min depth is 2.0 at T=0 -> Green

    expected = jp.array([[0, 0, 1], [0, 1, 0]])
    result = renderer.select_colors(depths, colors)
    assert jp.array_equal(result, expected)


# --- Integration Tests with Scenes ---


@pytest.fixture
def small_image_shape():
    return 32, 32


@pytest.fixture
def camera_pose():
    return SE3.view_transform(
        jp.array([0.0, 0.0, 5.0]),  # Camera at +5 Z
        jp.array([0.0, 0.0, 0.0]),  # Looking at origin
        jp.array([0.0, 1.0, 0.0]),
    )


@pytest.fixture
def rays(small_image_shape, camera_pose):
    return paz.graphics.camera.build_rays(small_image_shape, jp.pi / 3.0, camera_pose)


def test_render_reflection_scene(small_image_shape, camera_pose, rays):
    mirror_material = Material(
        color=jp.array([1.0, 1.0, 1.0]),
        reflective=0.8,
    )
    glass_material = Material(
        color=jp.array([0.9, 0.9, 1.0]),
        refractive=0.9,
        refractive_index=1.5,
    )
    floor_material = Material(color=jp.array([0.5, 0.5, 0.5]))

    floor = Plane(jp.eye(4), floor_material)
    sphere = Sphere(SE3.translation(jp.array([-1.5, 0.0, 0.0])), mirror_material)
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
    rays_shadow = paz.graphics.camera.build_rays(small_image_shape, jp.pi / 3.0, camera_pose_shadow)
    
    wall_transform = SE3.rotation_x(jp.pi/2) # Normal becomes z?
    wall = Plane(wall_transform, Material(color=jp.array([1.0, 1.0, 1.0])))
    
    blocker = Sphere(SE3.translation(jp.array([0.0, 0.0, 2.0])) @ SE3.scaling(jp.full(3, 0.5)))
    
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 0.0, 5.0]))]
    
    scene_blocked = Scene([wall, blocker])
    
    img_shadows_on, _ = renderer.render(
        small_image_shape, camera_pose_shadow, rays_shadow, scene_blocked, lights, mask=None, shadows=True
    )
    img_shadows_off, _ = renderer.render(
        small_image_shape, camera_pose_shadow, rays_shadow, scene_blocked, lights, mask=None, shadows=False
    )
    
    assert not jp.array_equal(img_shadows_on, img_shadows_off)


def test_render_shadow_mask(small_image_shape, camera_pose, rays):
    camera_pose_shadow = SE3.view_transform(
        jp.array([0.0, 0.0, 10.0]),
        jp.array([0.0, 0.0, 0.0]),
        jp.array([0.0, 1.0, 0.0]),
    )
    rays_shadow = paz.graphics.camera.build_rays(small_image_shape, jp.pi / 3.0, camera_pose_shadow)
    
    wall_transform = SE3.rotation_x(jp.pi/2)
    wall = Plane(wall_transform, Material(color=jp.array([1.0, 1.0, 1.0])))
    blocker = Sphere(SE3.translation(jp.array([0.0, 0.0, 2.0])))
    
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([5.0, 5.0, 5.0]))] 
    
    scene = Scene([wall, blocker])
    
    shadow_mask = jp.array([True, False])
    
    img_no_cast, _ = renderer.render(
        small_image_shape, camera_pose_shadow, rays_shadow, scene, lights, mask=None, shadows=True, shadow_mask=shadow_mask
    )
    
    img_cast, _ = renderer.render(
        small_image_shape, camera_pose_shadow, rays_shadow, scene, lights, mask=None, shadows=True, shadow_mask=None
    )
    
    assert not jp.array_equal(img_no_cast, img_cast)


def test_render_masked_objects(small_image_shape, camera_pose, rays):
    sphere = Sphere(SE3.translation(jp.array([0.0, 0.0, 0.0])))
    scene = Scene([sphere])
    lights = [PointLight(jp.array([1.0, 1.0, 1.0]), jp.array([0.0, 0.0, 5.0]))]
    
    mask = jp.array([False])
    
    img_hidden, _ = renderer.render(
        small_image_shape, camera_pose, rays, scene, lights, mask=mask, shadows=False
    )
    
    assert jp.all(img_hidden == 1.0)
    
    mask = jp.array([True])
    img_visible, _ = renderer.render(
        small_image_shape, camera_pose, rays, scene, lights, mask=mask, shadows=False
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
    rays_back = paz.graphics.camera.build_rays(small_image_shape, jp.pi/3.0, camera_pose_back)

    mirror_mat = Material(color=jp.array([0.0, 0.0, 0.0]), reflective=1.0, diffuse=0.0, ambient=0.0)
    red_mat = Material(color=jp.array([1.0, 0.0, 0.0]), ambient=1.0) 
    
    # Mirror at 0. Scale 2.0 to ensure hit.
    mirror = Sphere(SE3.scaling(jp.full(3, 2.0)), mirror_mat)
    # Red Obj at 10 (behind camera).
    red_obj = Sphere(SE3.translation(jp.array([0.0, 0.0, 10.0])) @ SE3.scaling(jp.full(3, 2.0)), red_mat)
    
    scene = Scene([mirror, red_obj])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 0.0, 5.0]))]
    
    shapes, lights_c, mask_c = paz.graphics.scene.compile(scene, lights, None)
    
    img_1b, _ = renderer._render_bounced(
        small_image_shape, camera_pose_back, rays_back, shapes, lights_c, mask_c, False, None, max_bounces=1
    )
    
    img_2b, _ = renderer._render_bounced(
        small_image_shape, camera_pose_back, rays_back, shapes, lights_c, mask_c, False, None, max_bounces=2
    )
    
    # 1 Bounce: Hit mirror. Mirror is black. Image should be black (0) where hit, white (1) background.
    # 2 Bounces: Hit mirror -> Reflect -> Hit Red. Image should be red.
    
    assert not jp.array_equal(img_1b, img_2b)
    
    # Also verify we hit something (not all white)
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