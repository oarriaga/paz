"""Mesh representation of the MANO hand model.
See `here <https://mano.is.tue.mpg.de/>`_ for details on the model.
Their code has been refactored and documented here from Dr. Alexander Fabisch / DFKI-RIC.
"""
import json
from scipy import sparse
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import numpy as np
from pkg_resources import resource_filename
import open3d as o3d


class HandState:
    """Holds an Open3D mesh representation of the Mano hand model.
    Mano is described by Romero (2017).
    J. Romero, D. Tzionas and M. J. Black:
    Embodied Hands: Modeling and Capturing Hands and Bodies Together (2017),
    ACM Transactions on Graphics, (Proc. SIGGRAPH Asia),
    https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/392/Embodied_Hands_SiggraphAsia2017.pdf
    website: https://mano.is.tue.mpg.de/
    Parameters
    ----------
    left : bool, optional (default: True)
        Left hand. Right hand otherwise.
    """

    def __init__(self, left=True):
        model_parameters = load_model(left)

        self.betas = np.zeros(10)
        self.pose = np.zeros(48)

        self.faces = model_parameters.pop("f")
        self.shape_parameters = {
            "v_template": model_parameters["v_template"],
            "J_regressor": model_parameters["J_regressor"],
            "shapedirs": model_parameters["shapedirs"],
        }
        self.pose_parameters = {
            "weights": model_parameters["weights"],
            "kintree_table": model_parameters["kintree_table"],
            "posedirs": model_parameters["posedirs"],
        }

        self.pose_parameters["J"], self.pose_parameters["v_template"] = \
            apply_shape_parameters(betas=self.betas, **self.shape_parameters)
        self.vertices = hand_vertices(pose=self.pose, **self.pose_parameters)

        self.material = o3d.visualization.rendering.MaterialRecord()
        color = np.array([245, 214, 175, 255]) / 255.0
        self.material.base_color = color
        self.material.shader = "defaultLit"

        self._mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.faces))
        self._mesh.compute_vertex_normals()
        self._mesh.paint_uniform_color(color[:3])

        self._points = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(self.vertices))
        self._points.paint_uniform_color((0, 0, 0))

        self.mesh_updated = False

    def set_pose_parameter(self, idx, value):
        self.pose[idx] = value
        self.recompute_shape()
        self.mesh_updated = True

    def set_shape_parameter(self, idx, value):
        self.betas[idx] = value
        self.recompute_shape()
        self.mesh_updated = True

    def recompute_shape(self):
        self.pose_parameters["J"], self.pose_parameters["v_template"] = \
            apply_shape_parameters(betas=self.betas, **self.shape_parameters)

    @property
    def n_pose_parameters(self):
        return self.pose.shape[0]

    @property
    def n_shape_parameters(self):
        return self.betas.shape[0]

    @property
    def hand_mesh(self):
        if self.mesh_updated:
            self.recompute_mesh()
            self.mesh_updated = False

        return self._mesh

    def recompute_mesh(self, mesh2world=None, vertex_normals=True,
                       triangle_normals=True):
        self.vertices[:, :] = hand_vertices(
            pose=self.pose, **self.pose_parameters)
        if mesh2world is not None:
            self.vertices[:, :] = pt.transform(
                mesh2world, pt.vectors_to_points(self.vertices))[:, :3]
        self._mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        if vertex_normals:
            self._mesh.compute_vertex_normals()
        if triangle_normals:
            self._mesh.compute_triangle_normals()
        self._points.points = o3d.utility.Vector3dVector(self.vertices)

    @property
    def hand_pointcloud(self):
        if self.mesh_updated:
            self.recompute_mesh()
            self.mesh_updated = False

        return self._points


def load_model(left=True):
    """Load model parameters.
    Parameters
    ----------
    left : bool, optional (default: True)
        Left hand. Right hand otherwise.
    Returns
    -------
    model_parameters : dict
        Parameters that we need to compute mesh of hand.
    """

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if left:
        filename = os.path.join(dir_path, "mano_left.json")
    else:
        filename = os.path.join(dir_path, "mano_right.json")

    with open(filename, "r") as f:
        model_kwargs = json.load(f)

    J_regressor = model_kwargs["J_regressor"]

    model_kwargs["J_regressor"] = sparse.csc_matrix(
        (J_regressor["data"], J_regressor["indices"], J_regressor["indptr"]))

    for k in ["f", "kintree_table", "J", "weights", "posedirs", "v_template",
              "shapedirs"]:
        model_kwargs[k] = np.array(model_kwargs[k])

    return model_kwargs


def apply_shape_parameters(v_template, J_regressor, shapedirs, betas):
    """Apply shape parameters.
    Parameters
    ----------
    v_template : array, shape (n_vertices, 3)
        Vertices of template model
    J_regressor : array, shape (n_parts, n_vertices)
        Joint regressor matrix that is used to predict joints for a body.
    shapedirs : array, shape (n_vertices, 3, n_principal_shape_parameters)
        Orthonormal principal components of shape displacements. Deviation
        from the template model.
    betas : array, shape (n_principal_shape_parameters,)
        Linear shape coefficients. These define the magnitude of deviation
        from the template model.
    Returns
    -------
    J : array, shape (n_parts, 3)
        Joint positions
    v_shaped : array, shape (n_vertices, 3)
        Shaped vertices of template model
    """
    v_shaped = v_template + shapedirs.dot(betas)
    return J_regressor.dot(v_shaped), v_shaped


def hand_vertices(J, weights, kintree_table, v_template, posedirs, pose=None):
    """Compute vertices of hand mesh.
    n_parts = 16
    n_vertices = 778
    n_principal_shape_parameters = 10
    Mesh shape is computed according to Loper et al. (2015).
    M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, M. J. Black:
    SMPL: A Skinned Multi-Person Linear Model (2015), ACM Transactions on
    Graphics (Proc. SIGGRAPH Asia), pp 248:1-248:16,
    http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf
    Parameters
    ----------
    J : array, shape (n_parts, 3)
        Joint positions
    weights : array, shape (n_vertices, n_parts)
        Blend weight matrix, how much does the rotation of each part effect
        each vertex
    kintree_table : array, shape (2, n_parts)
        Table that describes the kinematic tree of the hand.
        kintree_table[0, i] contains the index of the parent part of part i
        and kintree_table[1, :] does not matter for the MANO model.
    v_template : array, shape (n_vertices, 3)
        Vertices of template model
    posedirs : array, shape (n_vertices, 3, 9 * (n_parts - 1))
        Orthonormal principal components of pose displacements.
    pose : array, shape (n_parts * 3)
        Hand pose parameters
    """
    if pose is None:
        pose = np.zeros(kintree_table.shape[1] * 3)
    pose = pose.reshape(-1, 3)
    v_posed = v_template + posedirs.dot(lrotmin(pose))
    vertices = forward_kinematic(pose, v_posed, J, weights, kintree_table)
    return vertices


try:
    from mano_fast import hand_vertices
except ImportError:
    pass  # using Python version


def lrotmin(p):
    """Compute offset magnitudes to the template model from pose parameters.
    Parameters
    ----------
    pose : array, shape (n_parts * 3)
        Hand pose parameters
    Returns
    -------
    offset_magnitudes : array, shape (135,)
        Magnitudes of offsets computed from pose parameters
    """
    return np.concatenate(
        [(pr.matrix_from_compact_axis_angle(np.array(pp)) - np.eye(3)).ravel()
         for pp in p[1:]]).ravel()


def forward_kinematic(pose, v, J, weights, kintree_table):
    """Computes the blending of joint influences for each vertex.
    Parameters
    ----------
    pose : array, shape (n_parts * 3)
        Hand pose parameters
    v : array, shape (n_vertices, 3)
        Vertices
    J : array, shape (n_parts, 3)
        Joint positions
    weights : array, shape (n_vertices, n_parts)
        Blend weight matrix, how much does the rotation of each part effect
        each vertex
    kintree_table : array, shape (2, n_parts)
        Table that describes the kinematic tree of the hand.
        kintree_table[0, i] contains the index of the parent part of part i
        and kintree_table[1, :] does not matter for the MANO model.
    Returns
    -------
    v : array, shape (n_vertices, 3)
        Transformed vertices
    """
    A = global_rigid_transformation(pose, J, kintree_table)
    T = A.dot(weights.T)

    rest_shape_h = np.vstack((v.T, np.ones((1, v.shape[0]))))

    v = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
         T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
         T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
         T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T

    return v[:, :3]


def global_rigid_transformation(pose, J, kintree_table):
    """Computes global rotation and translation of the model.
    Parameters
    ----------
    pose : array, shape (n_parts * 3)
        Hand pose parameters
    J : array, shape (n_parts, 3)
        Joint positions
    kintree_table : array, shape (2, n_parts)
        Table that describes the kinematic tree of the hand.
        kintree_table[0, i] contains the index of the parent part of part i
        and kintree_table[1, :] does not matter for the MANO model.
    Returns
    -------
    A : array, shape (4, 4, n_parts)
        Transformed joint poses
    """
    id_to_col = {kintree_table[1, i]: i
                 for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]]
              for i in range(1, kintree_table.shape[1])}

    results = {0: pt.transform_from(
        pr.matrix_from_compact_axis_angle(pose[0, :]), J[0, :])}
    for i in range(1, kintree_table.shape[1]):
        T = pt.transform_from(pr.matrix_from_compact_axis_angle(
            pose[i, :]), J[i, :] - J[parent[i], :])
        results[i] = results[parent[i]].dot(T)

    results = [results[i] for i in sorted(results.keys())]

    def pack(x):
        return np.hstack([np.zeros((4, 3)), x.reshape((4, 1))])

    return np.dstack(
        [results[i] - pack(results[i].dot(np.hstack(((J[i, :]), (0,)))))
         for i in range(len(results))])