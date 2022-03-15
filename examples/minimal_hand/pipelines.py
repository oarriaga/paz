# from minimal_hand_estimator import MinimalHandEstimator
from paz.abstract import Processor

from paz import processors as pr
import processors as pe
from paz.backend.image import flip_left_right
from paz.backend.image import draw_circle

import numpy as np
import pytransform3d.transformations as pyt
import pytransform3d.rotations as pyr

from wrappers import ModelPipeline
from kinematics import mpii_to_mano, MANOHandJoints, mano_to_mpii
from hand_mesh import HandMesh


class MANOHandPoseEstimation(Processor):
    def __init__(self, size=128):
        super(MANOHandPoseEstimation).__init__()
        self.size = size
        self.hand_estimator = MinimalHandEstimator()
        self.resize_img = pr.ResizeImage((size, size))
        self.draw_skeleton = pe.DrawHandSkeleton('hand')
        self.wrap = pr.WrapOutput(
            ['joint_angles', 'absolute_joint_angle_quaternions', 'image',
             'global_pos_joints', 'uv'])

    def call(self, input_image, flip_input_image=False):
        H, W = input_image.shape[:2]
        scale = np.array([H/self.size, W/self.size])
        image = self.resize_img(input_image)
        if flip_input_image:
            image = flip_left_right(image)

        joint_angles, absolute_joint_angle_quaternions, global_pos_joints, uv = \
            self.hand_estimator.predict(image)

        uv = np.flip(uv, axis=1)
        uv = [uv*scale * 4]
        image = self.draw_skeleton(input_image, uv)
        return self.wrap(joint_angles, absolute_joint_angle_quaternions,
                         image, global_pos_joints, uv)


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

    def predict(self, image):
        """ Detects mano_params from images.

        # Arguments
            image: image shape (128,128)
        """

        global_pos_joints, theta_mpii, uv = self.model.process(image)
        absolute_angle_quaternions = self.mpii_to_mano(theta_mpii)  # quaternions
        joint_angles = self.calculate_handstate_joint_angles_from_min_hand_absolute_angles(absolute_angle_quaternions)

        global_pos_joints = self.mpii_to_mano(global_pos_joints)  # quaternions

        return joint_angles, absolute_angle_quaternions, global_pos_joints, uv

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
            t_posed_super_rotated[i] = pyt.transform_from(
                pyr.matrix_from_quaternion(quats[i]),
                J[i]
            )

        t_relative = np.zeros(shape=(21, 3, 3))

        # For each quaternion Q:
        for i in range(len(quats)):

            # Calc transformation with inverted rotation of Qi
            T_abs_rotations_i_inverted = pyt.invert_transform(
                pyt.transform_from(
                    pyr.matrix_from_quaternion(quats[i]),
                    [0, 0, 0]  # translation does not matter
                )
            )

            # Update Q_orientation if joint i has a parent (substract parents orientation)
            parent_index = MANOHandJoints.parents[i]
            if parent_index is not None:
                # Concatenate transformation get rotation difference (child to parent):
                # posed and super rotated joint i
                # inverted rotation of Qi
                t_posed_rotation_child_to_parent_i = pyt.concat(
                    t_posed_super_rotated[parent_index],
                    T_abs_rotations_i_inverted
                )

                # clear out translationand keep only rotation
                t = pyt.pq_from_transform(t_posed_rotation_child_to_parent_i)
                t_rotation_child_to_parent_i = np.array([t[3], t[4], t[5], t[6]])

                t_relative[i] = pyr.matrix_from_quaternion(
                    pyr.q_conj(t_rotation_child_to_parent_i)
                )

        # Generate final array with 16 joint angles
        joint_angles = np.zeros(shape=(21, 3))

        # Root joint gets same orientation like absolute root quaternion
        joint_angles[0] = pyr.compact_axis_angle_from_matrix(
            pyr.matrix_from_quaternion(quats[0])
        )

        # Map of childs array_index = joint_index => parent_joint_index
        childs = [
            [1, 4, 7, 10, 13],  # root_joint has multiple childs
            2, 3, 16, 5, 6, 17, 8, 9, 18, 11, 12, 19, 14, 15, 20  # other joints have exactly one parent
        ]
        # Joint 1-15 gets calculated orientation of child's join
        for i in range(1, 16):
            joint_angles[i] = pyr.compact_axis_angle_from_matrix(
                t_relative[childs[i]]
            )

        return joint_angles

    """
    Rotate reference joints by estimated absolute quats
    """

    def rotated_ref_joints_from_quats(self, quat):
        rotation_matrices = np.zeros(shape=(21, 3, 3))
        for j in range(len(quat)):
            rotation_matrices[j] = pyr.matrix_from_quaternion(quat[j])
        mats = np.stack(rotation_matrices, 0)
        joint_xyz = np.matmul(mats, self.hand_mesh.ref_pose)[..., 0]
        return joint_xyz