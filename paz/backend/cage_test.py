import jax.numpy as jp
import numpy as np
import paz
import trimesh


def build_test_cage():
    mesh = trimesh.creation.icosphere(2, 1.0)
    vertices = jp.array(mesh.vertices.view(np.ndarray))
    faces = jp.array(mesh.faces.view(np.ndarray))
    return vertices, faces


def test_compute_mesh_weights_returns_one_row_per_query():
    cage_vertices, cage_faces = build_test_cage()
    queries = jp.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    weights = paz.cage.compute_mesh_weights(queries, cage_vertices, cage_faces)
    assert weights.shape == (2, len(cage_vertices))


def test_control_mesh_recovers_queries_from_same_cage():
    cage_vertices, cage_faces = build_test_cage()
    queries = jp.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    weights = paz.cage.compute_mesh_weights(queries, cage_vertices, cage_faces)
    reconstructed = paz.cage.control_mesh(weights, cage_vertices)
    assert jp.allclose(reconstructed, queries, atol=1e-4)


def test_paz_exports_cage_backend():
    cage_vertices, cage_faces = build_test_cage()
    query = jp.array([0.0, 0.0, 0.0])
    weights = paz.cage.compute_mean_value_coordinate(
        cage_vertices, cage_faces, query)
    assert weights.shape == (len(cage_vertices),)
