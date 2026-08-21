import jax.numpy as jp

import paz
from paz.backend.lie import SE3
from paz.graphics.types import Material, Mesh, PointLight
from paz.graphics.mesh.builders import build_cube
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


def test_render_coordinates_shapes_and_object_frame_bounds():
    mesh = make_single_mesh()
    pose = camera_looking_at_origin()
    args = (20, 20), jp.pi / 4, pose, mesh, 1024
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
    render_args = (20, 20), jp.pi / 4, pose, scene, None, lights
    _, depth = paz.graphics.render(*render_args, (1, 1), 1024)
    _, hit = render_coordinates((20, 20), jp.pi / 4, pose, mesh, 1024)
    assert jp.array_equal(hit, depth > 0)
