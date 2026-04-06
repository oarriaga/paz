import numpy as np
import jax.numpy as jp
import paz
from paz.backend import draw


def make_camera_matrix():
    return jp.array(
        [
            [32.0, 0.0, 32.0, 0.0],
            [0.0, 32.0, 32.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )


def make_transform():
    return jp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def make_mesh():
    vertices, faces, edges = paz.graphics.mesh.build_sphere(1.0, 0)
    vertices = jp.expand_dims(vertices, axis=0)
    faces = jp.expand_dims(faces, axis=0)
    edges = jp.expand_dims(edges, axis=0)
    transform = jp.expand_dims(make_transform(), axis=0)
    colors = jp.ones_like(vertices)
    mesh_args = vertices, colors, transform, None, faces, edges
    return paz.graphics.mesh.Mesh(*mesh_args)


def test_boxes_accepts_shared_color():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    boxes = jp.array([[5, 5, 20, 20], [30, 30, 50, 50]])
    result = paz.draw.boxes(image, boxes, (10, 20, 30), 2)
    expected = np.array([10, 20, 30], dtype=np.uint8)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(np.all(result == expected, axis=-1))


def test_boxes_accepts_one_color_per_box():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    boxes = jp.array([[5, 5, 20, 20], [30, 30, 50, 50]])
    colors = [(10, 20, 30), (40, 50, 60)]
    result = paz.draw.boxes(image, boxes, colors, 2)
    color_A = np.array([10, 20, 30], dtype=np.uint8)
    color_B = np.array([40, 50, 60], dtype=np.uint8)
    assert np.any(np.all(result == color_A, axis=-1))
    assert np.any(np.all(result == color_B, axis=-1))


def test_mesh2D_preserves_shape_and_dtype():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    vertices2D = jp.array([[20, 20], [20, 40], [40, 40], [40, 20]])
    edges = jp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    args = image, vertices2D, edges, (10, 20, 30)
    result = paz.draw.mesh2D(*args, edge_scale=0.5)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(result != 0)


def test_poses_preserves_shape_and_dtype():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    transforms = jp.stack([make_transform()])
    camera_matrix = make_camera_matrix()
    result = paz.draw.poses(image, transforms, camera_matrix)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(result != 0)


def test_poses_accepts_explicit_colors():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    transforms = jp.stack([make_transform()])
    camera_matrix = make_camera_matrix()
    colors = ((10, 20, 30),)
    result = paz.draw.poses(image, transforms, camera_matrix, colors=colors)
    expected = np.array([10, 20, 30], dtype=np.uint8)
    assert np.any(np.all(result == expected, axis=-1))


def test_mesh_poses_preserves_shape_and_dtype():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    meshes = make_mesh()
    camera_matrix = make_camera_matrix()
    result = paz.draw.mesh_poses(image, meshes, camera_matrix)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(result != 0)


def test_mesh_poses_accepts_explicit_colors():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    meshes = make_mesh()
    camera_matrix = make_camera_matrix()
    args = image, meshes, camera_matrix, 2, 4, ((10, 20, 30),)
    result = paz.draw.mesh_poses(*args)
    expected = np.array([10, 20, 30], dtype=np.uint8)
    assert np.any(np.all(result == expected, axis=-1))


def test_poses_and_mesh_poses_compose():
    image = jp.zeros((64, 64, 3), dtype=jp.uint8)
    meshes = make_mesh()
    camera_matrix = make_camera_matrix()
    transforms = meshes.transform
    image = paz.draw.poses(image, transforms, camera_matrix)
    image = paz.draw.mesh_poses(image, meshes, camera_matrix)
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert np.any(image != 0)


def test_lincolors_uses_cache():
    before = draw._lincolors.cache_info().hits
    draw._lincolors(3)
    draw._lincolors(3)
    after = draw._lincolors.cache_info().hits
    assert after == before + 1
