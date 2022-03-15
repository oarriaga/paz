import numpy as np
import open3d as o3d
from mano_handstate import HandState


class ManoVisualizer:

    def __init__(self):
        self.hand_state = HandState(left=True)

    def recompute_mano_hand_state(self, relative_joint_rotations, mano_shape_betas=None):
        if mano_shape_betas is not None:
            self.hand_state.betas = mano_shape_betas

        self.hand_state.pose = np.ravel(relative_joint_rotations[:16])
        self.hand_state.recompute_shape()
        self.hand_state.recompute_mesh(mesh2world=mesh2world)
        # hand_state.vertices = hand_state.vertices * 3

        rgba = [145, 114, 255, 255]
        material = o3d.visualization.rendering.MaterialRecord()
        color = np.array(rgba) / 255.0
        material.base_color = color
        material.shader = "defaultLit"
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.hand_state.vertices),
            o3d.utility.Vector3iVector(self.hand_state.faces)
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color[:3])

        return mesh
