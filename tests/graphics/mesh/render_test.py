import jax.numpy as jp

import paz
from paz.backend.lie import SE3
from paz.graphics.types import Material, Mesh, PointLight
from paz.graphics.mesh.builders import build_cube
from paz.graphics.mesh.intersect import intersect_mesh
from paz.graphics.mesh.render import render_coordinates


def make_single_mesh():
    vertices, faces, edges = build_cube(1.0)
    color = jp.array([[0.7, 0.3, 0.1]])
    vertex_colors = jp.repeat(color, len(vertices), axis=0)
    transform = SE3.translation(jp.zeros(3))
    material = Material(jp.zeros(3), 0.1, 0.9, 0.1, 100)
    args = vertices, vertex_colors, transform, material, faces, edges
    return Mesh(*args)


def camera_looking_at_origin():
    camera_origin = jp.array([0.0, 1.0, -1.5])
    world_up = jp.array([0.0, 0.0, 1.0])
    return SE3.view_transform(camera_origin, jp.zeros(3), world_up)


def compute_object_frame_points(shape, intrinsics, pose, mesh):
    rays = paz.graphics.camera.build_rays(shape, intrinsics, pose)
    origins, directions = rays
    depth = intersect_mesh(mesh, origins, directions, 1024)[1]
    world_to_shape = jp.linalg.inv(mesh.transform)
    args = world_to_shape, origins, directions
    origins, directions = paz.algebra.transform_rays(*args)
    points = origins + jp.expand_dims(depth, -1) * directions
    return jp.reshape(points, (*shape, 3))


def test_render_coordinates_match_object_frame_hit_points():
    mesh = make_single_mesh()
    pose = camera_looking_at_origin()
    shape = (20, 20)
    intrinsics = paz.graphics.camera.compute_intrinsics(jp.pi / 4, *shape)
    coordinates, hit = render_coordinates(shape, intrinsics, pose, mesh, 1024)
    points = compute_object_frame_points(shape, intrinsics, pose, mesh)
    assert jp.allclose(coordinates[hit], points[hit], atol=1e-5)


def test_render_coordinates_shapes_and_object_frame_bounds():
    mesh = make_single_mesh()
    pose = camera_looking_at_origin()
    intrinsics = paz.graphics.camera.compute_intrinsics(jp.pi / 4, 20, 20)
    args = (20, 20), intrinsics, pose, mesh, 1024
    coordinates, hit = render_coordinates(*args)
    assert coordinates.shape == (20, 20, 3)
    assert hit.shape == (20, 20)
    assert jp.any(hit)
    inside = coordinates[hit]
    assert jp.all(inside >= -0.5 - 1e-4) and jp.all(inside <= 0.5 + 1e-4)
    assert jp.allclose(coordinates[~hit], 0.0)


def test_render_coordinates_mask_matches_render_depth():
    mesh = make_single_mesh()
    pose = camera_looking_at_origin()
    scene = paz.graphics.Scene([mesh])
    lights = [PointLight(jp.full((3,), 10.0), jp.array([0.0, 1.0, -1.5]))]
    intrinsics = paz.graphics.camera.compute_intrinsics(jp.pi / 4, 20, 20)
    render_args = (20, 20), intrinsics, pose, scene, None, lights
    _, depth = paz.graphics.render(*render_args, (1, 1), 1024)
    _, hit = render_coordinates((20, 20), intrinsics, pose, mesh, 1024)
    assert jp.array_equal(hit, depth > 0)
