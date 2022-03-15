import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

from wrappers import ModelPipeline
from kinematics import mpii_to_mano, MANOHandJoints, mano_to_mpii
from hand_mesh import HandMesh

class MinimalHandEstimator(object):
    """Mano Param detector.
    """

    def __init__(self):
        self.model = ModelPipeline(left=True) #Attention, needs to be left=True! Because pretrained models are
        #trained for left-hand-only

        #load hand_mesh to use its ref_pose
        self.hand_mesh = HandMesh(left=True) #Attention, decides which ref_pose is loaded! should be left here
        #because of pretrained models support only left hand

        self.mpii_to_mano = mpii_to_mano

    def get_vector_length_wrist_to_middle_mcp(self):
        wrist_joint = self.hand_mesh.joints[0]
        middle_mcp_joint = self.hand_mesh.joints[4]
        vec_len = np.linalg.norm(np.array(middle_mcp_joint - wrist_joint))
        return vec_len

    def get_media_pipe_tp_mpii_factor(self, media_pipe_wrist_to_middle_mc):
        mpii_wrist_to_middle_mc = self.get_vector_length_wrist_to_middle_mcp()
        return mpii_wrist_to_middle_mc / media_pipe_wrist_to_middle_mc

    def scale_media_pipe_to_mppi_ref_size(self, media_pipe_joints):
        wrist_joint = media_pipe_joints[0]
        middle_mcp_joint = media_pipe_joints[4]
        media_pipe_wrist_to_middle_mc = np.array(middle_mcp_joint - wrist_joint)
        vec_len = np.linalg.norm(np.array(media_pipe_wrist_to_middle_mc))
        return media_pipe_joints * self.get_media_pipe_tp_mpii_factor(vec_len)

    @staticmethod
    def flip_left_hand_joint_rotations_to_right_hand(left_hand_joint_rotations):
        right_hand_joint_rotations = np.zeros(shape=(16, 3))
        for i in range(16):
            right_hand_joint_rotations[i] = left_hand_joint_rotations[i]
            right_hand_joint_rotations[i][2] = left_hand_joint_rotations[i][2] * -1
            right_hand_joint_rotations[i][1] = left_hand_joint_rotations[i][1] * -1
        return right_hand_joint_rotations

    def predict(self, image):
        """ Detects mano_params from images.

        # Arguments
            image: image shape (128,128)
        """

        global_pos_joints, theta_mpii = self.model.process(image)
        absolute_angle_quaternions = self.mpii_to_mano(theta_mpii)  # quaternions
        joint_angles = self.calculate_handstate_joint_angles_from_min_hand_absolute_angles(absolute_angle_quaternions)

        global_pos_joints = self.mpii_to_mano(global_pos_joints)  # quaternions

        return joint_angles, absolute_angle_quaternions, global_pos_joints

    def predict_from_3d_joints(self, xyz):

        xyz = mano_to_mpii(xyz)
        global_pos_joints, theta_mpii = self.model.process_3d_joints(xyz)
        absolute_angle_quaternions = self.mpii_to_mano(theta_mpii)  # quaternions
        joint_angles = self.calculate_handstate_joint_angles_from_min_hand_absolute_angles(absolute_angle_quaternions)

        return joint_angles, absolute_angle_quaternions

    """
    Convert absolute joint angles to relative joint angles
    """
    def calculate_handstate_joint_angles_from_min_hand_absolute_angles(self, quats):

        # rotate reference joints and get posed hand sceleton J
        J = self.rotated_ref_joints_from_quats(quats)

        # combine each joint with absolute rotation to transformation: t_posed_super_rotated
        t_posed_super_rotated = np.zeros(shape=(21, 4, 4))
        for i in range(21):
            t_posed_super_rotated[i] = pt.transform_from(
                pr.matrix_from_quaternion(quats[i]),
                J[i]
            )

        t_relative = np.zeros(shape=(21, 3, 3))

        # For each quaternion Q:
        for i in range(len(quats)):

            # Calc transformation with inverted rotation of Qi
            T_abs_rotations_i_inverted = pt.invert_transform(
                pt.transform_from(
                    pr.matrix_from_quaternion(quats[i]),
                    [0, 0, 0]  # translation does not matter
                )
            )

            # Update Q_orientation if joint i has a parent (substract parents orientation)
            parent_index = MANOHandJoints.parents[i]
            if parent_index is not None:
                # Concatenate transformation get rotation difference (child to parent):
                # posed and super rotated joint i
                # inverted rotation of Qi
                t_posed_rotation_child_to_parent_i = pt.concat(
                    t_posed_super_rotated[parent_index],
                    T_abs_rotations_i_inverted
                )

                # clear out translationand keep only rotation
                t = pt.pq_from_transform(t_posed_rotation_child_to_parent_i)
                t_rotation_child_to_parent_i = np.array([t[3], t[4], t[5], t[6]])

                t_relative[i] = pr.matrix_from_quaternion(
                    pr.q_conj(t_rotation_child_to_parent_i)
                )

        # Generate final array with 16 joint angles
        joint_angles = np.zeros(shape=(21, 3))

        # Root joint gets same orientation like absolute root quaternion
        joint_angles[0] = pr.compact_axis_angle_from_matrix(
            pr.matrix_from_quaternion(quats[0])
        )

        # Map of childs array_index = joint_index => parent_joint_index
        childs = [
            [1, 4, 7, 10, 13],  # root_joint has multiple childs
            2, 3, 16, 5, 6, 17, 8, 9, 18, 11, 12, 19, 14, 15, 20  # other joints have exactly one parent
        ]
        # Joint 1-15 gets calculated orientation of child's join
        for i in range(1, 16):
            joint_angles[i] = pr.compact_axis_angle_from_matrix(
                t_relative[childs[i]]
            )

        return joint_angles

    """
    Rotate reference joints by estimated absolute quats
    """

    def rotated_ref_joints_from_quats(self, quat):
        rotation_matrices = np.zeros(shape=(21, 3, 3))
        for j in range(len(quat)):
            rotation_matrices[j] = pr.matrix_from_quaternion(quat[j])
        mats = np.stack(rotation_matrices, 0)
        joint_xyz = np.matmul(mats, self.hand_mesh.ref_pose)[..., 0]
        return joint_xyz