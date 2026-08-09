import pytest
import jax
import jax.numpy as jp

import paz
from paz import SE3
from paz.graphics.renderer import intersect, shadow
from paz.graphics.shapes.sphere import intersect_canonical_sphere
from paz.graphics.types import Material, PointLight, Sphere, Plane, Scene


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
    scene = Scene([sphere, Plane()])
    lights = [PointLight(jp.ones(3), jp.array([0.0, 3.0, -3.0]))]
    return scene, lights


def compute_selected_shadow_depths(camera_pose, image_shape=(120, 160)):
    scene, lights = build_shadow_scene()
    rays = paz.graphics.camera.build_rays(image_shape, jp.pi / 3.0, camera_pose)
    compiled = paz.graphics.scene.compile(scene, lights, None)
    shapes, mask, lights = compiled.shapes, compiled.mask, compiled.lights
    candidates = intersect.build_candidates(compiled, rays, None)
    closest = intersect.find_closest(*candidates)
    vector = lights[0].position - closest.point
    distance = jp.squeeze(paz.algebra.compute_norms(vector, 1), axis=1)
    light_directions = vector / jp.expand_dims(distance, 1)
    origins = shadow.compute_shadow_ray_origins(closest.point, closest.normal)
    group_args = shapes, origins, light_directions
    intersections = intersect.intersect_shadow_groups(*group_args)
    hit_masks, depths, _, _, _, shape_indices = intersections
    transparencies = jp.array([shape.material.transparency for shape in shapes])
    shadow_masks = jp.where(jp.expand_dims(mask, 1), hit_masks, False)
    is_transparent = jp.expand_dims(transparencies > 0.0, 1)
    shadow_masks = jp.where(is_transparent, False, shadow_masks)
    select_args = shadow_masks, depths, shape_indices, closest.primitive_index
    select_args += closest.normal, light_directions
    shadow_masks, depths = shadow.select_shadow_depths(*select_args)
    return shadow_masks, depths, shape_indices, closest.primitive_index


def take_shape_depths(depths, shape_indices, receiver_indices, shape_index):
    shape_row = int(jp.argwhere(shape_indices == shape_index)[0, 0])
    receiver_mask = receiver_indices == shape_index
    return depths[shape_row][receiver_mask]


def test_compute_shadow_ray_origins_avoid_lit_side_self_hit():
    points = jp.array([[0.0, 1.0, 0.0]])
    normals = jp.array([[0.0, 1.0, 0.0]])
    light_position = jp.array([[0.0, 3.0, -3.0]])
    directions = paz.algebra.normalize(light_position - points)
    origins = shadow.compute_shadow_ray_origins(points, normals)
    hit_mask, _, _ = intersect_canonical_sphere(origins, directions)
    assert not bool(hit_mask[0])


def test_compute_surface_points_offset_hit():
    point = jp.array([[0.0, 0.0, 0.0]])
    normal = jp.array([[0.0, 0.0, -1.0]])
    over_point, under_point = shadow.compute_surface_points(point, normal)
    assert over_point[0, 2] < -(shadow.SHADOW_ORIGIN_EPSILON / 2.0)
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
    args = hit_masks, depths, shape_indices, receiver_indices
    args += receiver_normals, light_directions
    hit_masks, depths = shadow.select_shadow_depths(*args)
    assert not bool(hit_masks[0, 0])
    assert float(depths[0, 0]) == pytest.approx(paz.graphics.FARAWAY)
    assert bool(hit_masks[1, 0])
    assert float(depths[1, 0]) == pytest.approx(5e-4)


def test_select_shadow_depths_keep_back_side_second_root():
    depths = jp.array(
        [
            [[1e-6], [0.2]],
            [[5e-4], [1000.0]],
        ]
    )
    hit_masks = jp.array([[True], [True]])
    shape_indices = jp.array([0, 1])
    receiver_indices = jp.array([0])
    receiver_normals = jp.array([[0.0, 1.0, 0.0]])
    light_directions = jp.array([[0.0, -1.0, 0.0]])
    args = hit_masks, depths, shape_indices, receiver_indices
    args += receiver_normals, light_directions
    hit_masks, depths = shadow.select_shadow_depths(*args)
    occlusion_args = hit_masks, depths, jp.array([0.01])
    result = shadow.compute_occlusion_mask(*occlusion_args)
    assert bool(hit_masks[0, 0])
    assert float(depths[0, 0]) == pytest.approx(0.2)
    assert bool(hit_masks[1, 0])
    assert float(depths[1, 0]) == pytest.approx(5e-4)
    assert float(result[0]) == 1.0


def test_compute_occlusion_mask():
    light_lengths = jp.array([10.0, 10.0, 10.0, 10.0])
    depths = jp.array(
        [[paz.graphics.FARAWAY, 5.0, 10.0, 15.0], [11.0, 11.0, 10.0, 11.0]]
    )
    rows = [[True, True, True, True], [False, False, True, False]]
    hit_masks = jp.array(rows)
    args = hit_masks, depths, light_lengths
    result = shadow.compute_occlusion_mask(*args)
    assert float(result[0]) == 0.0
    assert float(result[1]) == 1.0
    assert float(result[2]) == 1.0
    assert float(result[3]) == 0.0


def test_saved_pose_sphere_self_shadow_keeps_later_roots():
    selected = compute_selected_shadow_depths(OLD_CAMERA_POSE)
    _, depths, shape_indices, receiver_indices = selected
    args = depths, shape_indices, receiver_indices, 0
    sphere_depths = take_shape_depths(*args)
    assert int(jp.sum(sphere_depths < 1e-2)) > 0
    assert float(jp.min(sphere_depths)) > shadow.SHADOW_SELF_HIT_EPSILON


def test_saved_pose_floor_self_hits_stay_filtered():
    selected = compute_selected_shadow_depths(NEW_CAMERA_POSE)
    _, depths, shape_indices, receiver_indices = selected
    plane_depths = take_shape_depths(depths, shape_indices, receiver_indices, 1)
    assert int(jp.sum(plane_depths < 1e-2)) == 0


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU only")
def test_gpu_saved_pose_floor_has_no_near_zero_self_hits():
    selected = compute_selected_shadow_depths(NEW_CAMERA_POSE, (240, 320))
    _, depths, shape_indices, receiver_indices = selected
    plane_depths = take_shape_depths(depths, shape_indices, receiver_indices, 1)
    assert int(jp.sum(plane_depths < 1e-2)) == 0
