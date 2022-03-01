import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr

from .wrappers import ModelPipeline
from .kinematics import mpii_to_mano, MANOHandJoints, mano_to_mpii
from .hand_mesh import HandMesh

class MinimalHandEstimator(object):
    """Mano Param detector.
    """

    def __init__(self):
        self.model = ModelPipeline(left=True) #Attetion: Needs to be left=True! Because pretrained models are
        #trained for left-hand-only

        #load hand_mesh to use its ref_pose
        self.hand_mesh = HandMesh(left=True) #Attetion, decides which ref_pose is loaded! should be left here
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
        # Calculate Minimal_Hand to MANO Transformation
        joint_orientations = np.zeros(shape=(21, 3, 3))

        # For each quaternion:
        # Transformation: rotation of ref_joint i + global position of rotated ref_joint i
        rotated_ref_joints_transformations = self.rotated_ref_joints_transformations(quats)

        # For each quaternion:
        for i in range(21):

            # Calc Transformation: inverted rotation of ref_joint i + ?,?,? translation
            joint_transformation_i_pos_no_matter = pt.transform_from(
                pr.matrix_from_quaternion(quats[i]),
                [999,999,999]
            )
            joint_transformation_i_pos_no_matter_inverted = pt.invert_transform(joint_transformation_i_pos_no_matter)

            # Concatenate transformation:
            #   A(absolute rotation and translation from ref_joint i to absolute_joint i)
            # + B(inverted absolute quaternion i + ?,?,? translation)
            T_joint = pt.concat(
                rotated_ref_joints_transformations[i],
                joint_transformation_i_pos_no_matter_inverted
            )

            # Compute position and quaternion from transformation matrix T_joint
            # Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
            Q_joint = pt.pq_from_transform(T_joint)

            # keep only orientation in var: Q_orientation
            Q_orientation = np.array([Q_joint[3], Q_joint[4], Q_joint[5], Q_joint[6]])

            # Update Q_orientation if joint i has a parent (substract parents orientation)
            parent_index = MANOHandJoints.parents[i]
            if parent_index is not None:

                # Concatenate transformation:
                #   A(absolute rotation and translation from parent's ref_joint to parent's absolute_joint)
                # + B(inverted absolute quaternion i + ?,?,? translation)
                T_joint_parent = pt.concat(
                    rotated_ref_joints_transformations[parent_index],
                    joint_transformation_i_pos_no_matter_inverted
                )

                # Compute position and quaternion from transformation matrix T_joint
                # Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
                Q_parent_joint = pt.pq_from_transform(T_joint_parent)
                Q_orientation_parent = np.array([Q_parent_joint[3], Q_parent_joint[4], Q_parent_joint[5], Q_parent_joint[6]])

                #Concatenate Q_orientation with inverted Q_orientation_parent
                Q_orientation = pr.concatenate_quaternions(
                    Q_orientation,
                    pr.q_conj(Q_orientation_parent)
                )

            joint_orientations[i] = pr.matrix_from_quaternion(Q_orientation)

        # Generate final array with 16 joint angles
        joint_angles = np.zeros(shape=(21, 3))

        # Map of childs array_index = joint_index => parent_joint_index
        childs = [
            [1,4,7,10,13], # root_joint has multiple childs
            2,3,16,5,6,17,8,9,18,11,12,19,14,15,20 #other joints have exactly one parent
        ]
        # Root joint gets same orientation like absolute root quaternion
        joint_orientations[0] = pr.matrix_from_quaternion(quats[0])
        joint_angles[0] = pr.compact_axis_angle_from_matrix(
            joint_orientations[0]
        )

        # Joint 1-15 gets calculated orientation of child's join
        for i in range(1,16):
            joint_angles[i] = pr.compact_axis_angle_from_matrix(
                joint_orientations[childs[i]]
            )

        return joint_angles

    def rotation_matrices_from_quats(self, quat):
        mats = np.zeros(shape=(21, 3, 3))
        for j in range(MANOHandJoints.n_joints):
            mats[j] = pr.matrix_from_quaternion(quat[j])
        return mats

    def ref_joints_from_quats(self, quat):
        mats = np.stack(self.rotation_matrices_from_quats(quat), 0)
        joint_xyz = np.matmul(mats, self.hand_mesh.ref_pose)[..., 0]
        return joint_xyz

    def joints_from_quats(self, quat):
        """
        Set absolute (global) rotation for the hand.

        Parameters
        ----------
        quat : np.ndarray, shape [J, 4]
          Absolute rotations for each joint in quaternion.

        Returns
        -------
        np.ndarray, joints [J, 3]
          Ref_pose rotated by quaternions
        """
        mats = np.stack(self.rotation_matrices_from_quats(quat), 0)
        pose = np.matmul(mats, self.hand_mesh.ref_pose)
        joint_xyz = [None] * MANOHandJoints.n_joints
        for j in range(MANOHandJoints.n_joints):
            joint_xyz[j] = pose[j]
            parent = MANOHandJoints.parents[j]
            if parent is not None:
                joint_xyz[j] += joint_xyz[parent]
        joint_xyz = np.stack(joint_xyz, 0)[..., 0]

        return joint_xyz

    def rotated_ref_joints_transformations(self, quats):
        J = self.ref_joints_from_quats(quats)
        A = np.zeros(shape=(21, 4, 4))
        for i in range(21):
            A[i] = pt.transform_from(
                pr.matrix_from_quaternion(quats[i]),
                J[i]
            )
        return A