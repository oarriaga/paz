import trimesh
import pyrender
import numpy as np
from paz.backend.groups import build_rotation_matrix_x


def load_mesh(filename):
    mesh = trimesh.load(filename)
    return mesh


def visualize(mesh, pose=None):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh, pose=pose)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def extract_mesh_pointcloud(mesh):
    mesh_pointcloud = np.array(mesh.vertices)
    return mesh_pointcloud


def sample_pointcloud(pointcloud, dropout=0.9):
    """
    # Arguments
        pointcloud: Array (num_points, 3).
        dropout: Float [0, 1] probability to drop out each point.
    """
    num_points = len(pointcloud)
    mask = np.random.rand(num_points) > dropout
    return pointcloud[mask]


def rotate_pointcloud(pointcloud, rotation_matrix):
    return np.matmul(rotation_matrix, pointcloud.T).T


def translate_pointcloud(pointcloud, translation):
    assert translation.shape == (1, 3)
    return pointcloud + translation


if __name__ == "__main__":
    from icp import iterative_closes_point
    mesh = load_mesh('/home/octavio/.keras/paz/datasets/ycb_video/035_power_drill/textured.obj')
    pointcloud_A = extract_mesh_pointcloud(mesh)

    # rotation_matrix = build_rotation_matrix_x(np.pi / 2.0)
    rotation_matrix = build_rotation_matrix_x(np.pi / (6))
    pointcloud_B = sample_pointcloud(pointcloud_A, dropout=0.75)
    pointcloud_B = rotate_pointcloud(pointcloud_B, rotation_matrix)
    # pointcloud_B = translate_pointcloud(pointcloud_B, np.array([[0.3, -0.05, 0.0]]))
    # pointcloud_B = translate_pointcloud(pointcloud_B, np.array([[0.05, -0.05, 0.01]]))
    # visualize(mesh)
    pointcloud_mesh_B = pyrender.Mesh.from_points(pointcloud_B, colors=[0, 0, 1])
    pointcloud_mesh_A = pyrender.Mesh.from_points(pointcloud_A)
    # visualize(pointcloud_mesh)

    scene = pyrender.Scene()
    scene.add(pointcloud_mesh_A)
    scene.add(pointcloud_mesh_B)
    pyrender.Viewer(scene, use_raymond_lighting=True)

    affine_transform, distances, arg = iterative_closes_point(
        pointcloud_B, pointcloud_A, tolerance=1e-9)

    scene = pyrender.Scene()
    scene.add(pointcloud_mesh_A)
    scene.add(pointcloud_mesh_B, pose=affine_transform)
    # scene.add(pointcloud_mesh_B)
    pyrender.Viewer(scene, use_raymond_lighting=True)
