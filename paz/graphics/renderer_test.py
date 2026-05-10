import pytest
import jax
import jax.numpy as jp
import paz
from paz import SE3
from paz.graphics import renderer
from paz.graphics.shapes.sphere import intersect_canonical_sphere
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


OLD_CAMERA_POSE = jp.array(
    [
        [0.060758888721466064, 0.9969050884246826, 0.04988674074411392, 0.0],
        [0.0, 0.0499790757894516, -0.9987502694129944, 0.0],
        [-0.998152494430542, 0.060682956129312515, 0.003036673180758953, -4.242640495300293],  # fmt: skip
        [0.0, 0.0, 0.0, 1.0],
    ]
)


NEW_CAMERA_POSE = jp.array(
    [
        [0.9902160167694092, 0.11968734860420227, 0.07174412161111832, 0.0],
        [0.0, 0.5141358971595764, -0.857708752155304, 0.0],
        [-0.13954311609268188, 0.8493169546127319, 0.5091056227684021, -4.242640495300293],  # fmt: skip
        [0.0, 0.0, 0.0, 1.0],
    ]
)


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


def build_legacy_rays(image_shape, y_fov, world_to_camera):
    H, W = image_shape[:2]
    aspect_ratio = paz.graphics.camera.compute_aspect_ratio(H, W)
    H_world, W_world = paz.graphics.camera.compute_image_sizes(
        y_fov, aspect_ratio
    )
    directions = paz.graphics.camera.build_ray_directions(H, W, H_world, W_world)
    origins = paz.graphics.camera.build_ray_origins(H, W)
    camera_to_world = jp.linalg.inv(world_to_camera)
    return paz.algebra.transform_rays(camera_to_world, origins, directions)


def compute_max_abs_difference(array_A, array_B):
    return float(jp.max(jp.abs(array_A - array_B)))


def assert_render_matches(actual, expected, atol=1e-4):
    actual_image, actual_depth = actual
    expected_image, expected_depth = expected
    assert compute_max_abs_difference(actual_image, expected_image) <= atol
    assert compute_max_abs_difference(actual_depth, expected_depth) <= atol


def compute_selected_shadow_depths(camera_pose, image_shape=(120, 160)):
    scene, lights = build_shadow_scene()
    rays = paz.graphics.camera.build_rays(image_shape, jp.pi / 3.0, camera_pose)
    shapes, mask, _, lights = paz.graphics.scene.compile(scene, lights, None)
    intersections = renderer.intersect(shapes, rays, mask)
    closest = renderer.gather_closest(*intersections)
    vector = lights[0].position - closest.point
    distance = jp.squeeze(paz.algebra.compute_norms(vector, 1), axis=1)
    light_directions = vector / jp.expand_dims(distance, 1)
    shadow_ray_origins = renderer.compute_shadow_ray_origins(
        closest.point, closest.normal
    )
    intersections = renderer.intersect_shadow_groups(
        shapes, shadow_ray_origins, light_directions
    )
    hit_masks, depths, _, _, _, shape_indices = intersections
    transparencies = jp.array([shape.material.transparency for shape in shapes])
    shadow_masks = jp.where(jp.expand_dims(mask, 1), hit_masks, False)
    shadow_masks = jp.where(
        jp.expand_dims(transparencies > 0.0, 1), False, shadow_masks
    )
    shadow_masks, depths = renderer.select_shadow_depths(
        shadow_masks,
        depths,
        shape_indices,
        closest.shape_idx,
        closest.normal,
        light_directions,
    )
    return shadow_masks, depths, shape_indices, closest.shape_idx


def take_shape_depths(depths, shape_indices, receiver_indices, shape_index):
    shape_row = int(jp.argwhere(shape_indices == shape_index)[0, 0])
    receiver_mask = receiver_indices == shape_index
    return depths[shape_row][receiver_mask]


# --- Helper Functions Unit Tests ---


def test_take_closest():
    array = jp.array(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]
    )
    indices = jp.array([0, 1, 0])
    expected = jp.array([[1, 1, 1], [5, 5, 5], [3, 3, 3]])
    result = renderer.take_closest(array, indices)
    assert jp.array_equal(result, expected)


def test_compute_soft_occlusion():
    light_lengths = jp.array([10.0, 10.0, 10.0, 10.0])
    depths = jp.array(
        [[paz.graphics.FARAWAY, 5.0, 10.0, 15.0], [11.0, 11.0, 10.0, 11.0]]
    )
    hit_masks = jp.array([[True, True, True, True], [False, False, True, False]])
    result = renderer.compute_soft_occlusion(
        hit_masks, depths, light_lengths, slope=10.0
    )
    assert float(result[1]) > 0.9
    assert float(result[2]) == pytest.approx(0.5)
    assert float(result[3]) == 0.0
    assert float(result[0]) == 0.0


def test_compute_new_rays_reflection_is_normalized():
    normal = jp.array([[0.0, 1.0, 0.0]])
    eye = jp.array([[0.0, 1.0, -1.0]])
    point = jp.array([[0.0, 0.0, 0.0]])
    transparancies = jp.array([0.0])
    reflectance = jp.array([1.0])
    _, direction = renderer.compute_new_rays(
        normal,
        eye,
        jp.array([1.0]),
        point,
        transparancies,
        reflectance,
    )
    norm = jp.linalg.norm(direction, axis=-1)
    assert jp.allclose(norm, 1.0, atol=1e-5)


def test_compute_new_rays_refraction_is_normalized():
    normal = jp.array([[0.0, 0.0, -1.0]])
    eye = jp.array([[0.0, 0.0, -1.0]])
    point = jp.array([[0.0, 0.0, 0.0]])
    transparancies = jp.array([1.0])
    reflectance = jp.array([0.0])
    _, direction = renderer.compute_new_rays(
        normal,
        eye,
        jp.array([1.0 / 1.5]),
        point,
        transparancies,
        reflectance,
    )
    norm = jp.linalg.norm(direction, axis=-1)
    assert jp.allclose(norm, 1.0, atol=1e-5)


def test_compute_surface_points_offset_hit():
    point = jp.array([[0.0, 0.0, 0.0]])
    normal = jp.array([[0.0, 0.0, -1.0]])
    over_point, under_point = renderer.compute_surface_points(point, normal)
    assert over_point[0, 2] < -(renderer.SHADOW_ORIGIN_EPSILON / 2.0)
    assert point[0, 2] > over_point[0, 2]
    assert under_point[0, 2] > 0.0


def test_select_shadow_depths_discard_front_side_same_shape_hits():
    depths = jp.array(
        [
            [[0.2], [0.4]],
            [[5e-4], [1000.0]],
        ]
    )
    hit_masks = jp.array([[True], [True]])
    shape_indices = jp.array([0, 1])
    receiver_indices = jp.array([0])
    receiver_normals = jp.array([[0.0, 1.0, 0.0]])
    light_directions = jp.array([[0.0, 1.0, 0.0]])
    hit_masks, depths = renderer.select_shadow_depths(
        hit_masks,
        depths,
        shape_indices,
        receiver_indices,
        receiver_normals,
        light_directions,
    )
    assert not bool(hit_masks[0, 0])
    assert float(depths[0, 0]) == pytest.approx(paz.graphics.FARAWAY)
    assert bool(hit_masks[1, 0])
    assert float(depths[1, 0]) == pytest.approx(5e-4)


def test_select_shadow_depths_keep_back_side_second_root():
    depths = jp.array(
        [
            [[1e-4], [0.2]],
            [[5e-4], [1000.0]],
        ]
    )
    hit_masks = jp.array([[True], [True]])
    shape_indices = jp.array([0, 1])
    receiver_indices = jp.array([0])
    receiver_normals = jp.array([[0.0, 1.0, 0.0]])
    light_directions = jp.array([[0.0, -1.0, 0.0]])
    hit_masks, depths = renderer.select_shadow_depths(
        hit_masks,
        depths,
        shape_indices,
        receiver_indices,
        receiver_normals,
        light_directions,
    )
    result = renderer.compute_soft_occlusion(hit_masks, depths, jp.array([0.01]))
    assert bool(hit_masks[0, 0])
    assert float(depths[0, 0]) == pytest.approx(0.2)
    assert bool(hit_masks[1, 0])
    assert float(depths[1, 0]) == pytest.approx(5e-4)
    assert float(result[0]) > 0.5


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
    assert state.color.shape == (num_rays, 3)
    assert state.throughput.shape == (num_rays, 3)
    assert jp.all(state.refractive_index == 1.0)
    assert state.depth.shape == (num_rays,)
    assert state.hit_mask.dtype == bool


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
    reflectivities, transparencies, refractivities = (
        renderer.get_material_properties(shapes, indices)
    )
    assert reflectivities[0] == 0.5
    assert transparencies[1] == 0.8
    assert refractivities[0] == 1.0


def test_accumulate_color():
    num_rays = 2
    colors = jp.zeros((num_rays, 3))
    throughput = jp.ones((num_rays, 3))
    active_mask = jp.array([True, False])
    intersected_colors = jp.ones((num_rays, 3))
    reflectivities = jp.zeros((num_rays,))
    transparencies = jp.zeros((num_rays,))
    result = renderer.accumulate_color(
        colors,
        throughput,
        active_mask,
        intersected_colors,
        reflectivities,
        transparencies,
    )
    assert jp.array_equal(result[0], jp.array([1.0, 1.0, 1.0]))
    assert jp.array_equal(result[1], jp.array([0.0, 0.0, 0.0]))


def test_compute_shadow_ray_origins_avoid_lit_side_self_hit():
    points = jp.array([[0.0, 1.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    light_position = jp.array([[0.0, 3.0, -3.0]])
    directions = paz.algebra.normalize(light_position - points)
    origins = renderer.compute_shadow_ray_origins(points, normals)
    hit_mask, _, _ = intersect_canonical_sphere(origins, directions)
    assert not bool(hit_mask[0])


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


def snapshot_path(filename):
    return f"paz/graphics/snapshots/{filename}"


def assert_snapshot(array, filename, atol=1e-4):
    paz.assert_snapshot(array, snapshot_path(filename), atol=atol)


def render_scene(image_shape, camera_pose, scene, lights, mask=None,
                 shadows=False, tiles=(1, 1), chunk_size=1024,
                 shadow_mask=None, num_bounces=1):
    args = image_shape, jp.pi / 3.0, camera_pose, scene, mask, lights
    args += tiles, chunk_size
    return renderer.render(*args, shadows, shadow_mask, num_bounces)


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


def test_saved_pose_sphere_self_shadow_keeps_later_roots():
    _, depths, shape_indices, receiver_indices = compute_selected_shadow_depths(
        OLD_CAMERA_POSE
    )
    sphere_depths = take_shape_depths(depths, shape_indices, receiver_indices, 0)
    assert int(jp.sum(sphere_depths < 1e-2)) > 0
    assert float(jp.min(sphere_depths)) > renderer.SHADOW_SELF_HIT_EPSILON


def test_saved_pose_floor_self_hits_stay_filtered():
    _, depths, shape_indices, receiver_indices = compute_selected_shadow_depths(
        NEW_CAMERA_POSE
    )
    plane_depths = take_shape_depths(depths, shape_indices, receiver_indices, 1)
    assert int(jp.sum(plane_depths < 1e-2)) == 0


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU only")
def test_gpu_saved_pose_floor_has_no_near_zero_self_hits():
    _, depths, shape_indices, receiver_indices = compute_selected_shadow_depths(
        NEW_CAMERA_POSE, image_shape=(240, 320)
    )
    plane_depths = take_shape_depths(depths, shape_indices, receiver_indices, 1)
    assert int(jp.sum(plane_depths < 1e-2)) == 0


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


def test_shape_renderer_returns_uint8_frame(small_image_shape, camera_pose):
    material = Material(color=jp.array([1.0, 0.0, 0.0]))
    sphere = Sphere(jp.eye(4), material)
    scene = Scene([sphere])
    render_frame = paz.graphics.shape_renderer(
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
