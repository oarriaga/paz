import jax
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


def build_torus_cage(scale):
    args = {"major_sections": 32, "minor_sections": 16}
    mesh = trimesh.creation.torus(1.0, 0.3, **args)
    vertices = jp.array(scale * mesh.vertices, jp.float32)
    return vertices, jp.array(mesh.faces)


def build_torus_points(scale):
    random_state = np.random.default_rng(0)
    major_angles, minor_angles = random_state.uniform(0, 2 * np.pi, (2, 200))
    radii = 0.27 * np.cos(np.linspace(0, 1, 200))
    rings = 1.0 + (radii * np.cos(minor_angles))
    x = rings * np.cos(major_angles)
    y = rings * np.sin(major_angles)
    z = radii * np.sin(minor_angles)
    return jp.array(scale * np.stack([x, y, z], axis=1), jp.float32)


def compute_rebuild_error(points, cage_vertices, cage_faces):
    weights = paz.cage.compute_mesh_weights(points, cage_vertices, cage_faces)
    rebuilt = paz.cage.control_mesh(weights, cage_vertices)
    return jp.max(jp.linalg.norm(rebuilt - points, axis=1))


def test_control_mesh_recovers_points_inside_torus():
    points = build_torus_points(1.0)
    error = compute_rebuild_error(points, *build_torus_cage(1.0))
    assert error < 1e-4


def test_control_mesh_recovers_points_inside_small_torus():
    points = build_torus_points(0.05)
    error = compute_rebuild_error(points, *build_torus_cage(0.05))
    assert error < 1e-5


def test_compute_mesh_weights_returns_finite_weights():
    cage_vertices, cage_faces = build_torus_cage(1.0)
    points = build_torus_points(1.0)
    weights = paz.cage.compute_mesh_weights(points, cage_vertices, cage_faces)
    assert jp.all(jp.isfinite(weights))


def test_compute_mean_value_coordinate_at_cage_vertex():
    cage_vertices, cage_faces = build_torus_cage(1.0)
    args = (cage_vertices, cage_faces, cage_vertices[7])
    weights = paz.cage.compute_mean_value_coordinate(*args)
    assert jp.allclose(weights, jax.nn.one_hot(7, len(cage_vertices)))


def test_control_mesh_recovers_points_on_cage_triangles():
    cage_vertices, cage_faces = build_torus_cage(1.0)
    corners = cage_vertices[cage_faces]
    centers = jp.mean(corners, axis=1)
    edge_midpoints = jp.mean(corners[:, :2], axis=1)
    points = jp.concatenate([centers, edge_midpoints])
    error = compute_rebuild_error(points, cage_vertices, cage_faces)
    assert error < 1e-6


def test_jit_compute_mesh_weights_matches_unjitted():
    cage_vertices, cage_faces = build_torus_cage(1.0)
    points = build_torus_points(1.0)
    args = (points, cage_vertices, cage_faces)
    weights = normalize(paz.cage.compute_mesh_weights(*args))
    jitted_weights = normalize(jax.jit(paz.cage.compute_mesh_weights)(*args))
    assert jp.allclose(weights, jitted_weights, atol=1e-5)


def normalize(weights):
    return weights / jp.sum(weights, axis=1, keepdims=True)


def test_compute_mesh_weights_stays_below_one_face_term_per_point():
    mesh = trimesh.creation.uv_sphere(1.0, (23, 23))
    cage_vertices = jp.array(mesh.vertices, jp.float32)
    cage_faces = jp.array(mesh.faces)
    random_state = np.random.default_rng(0)
    points = random_state.uniform(-0.5, 0.5, (30000, 3))
    args = (jp.array(points, jp.float32), cage_vertices, cage_faces)
    compute = jax.jit(paz.cage.compute_mesh_weights)
    memory = compute.lower(*args).compile().memory_analysis()
    assert memory.temp_size_in_bytes < 30000 * len(cage_faces) * 4
    assert jp.all(jp.isfinite(compute(*args)))
