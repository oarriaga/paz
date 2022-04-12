import numpy as np
import processors as pe
from paz import processors as pr
from paz.backend.image import flip_left_right
from models.detnet import DetNet
# from models.detnet_tf2 import DetNet
from models.iknet import ModelIK


from backend import xyz_to_delta
from backend import mano_to_mpii
from backend import mpii_to_mano
from backend import MPIIHandJoints
from backend import load_json
from backend import calculate_handstate_joint_angles_from_min_hand_absolute_angles
from config import *


# class HandPoseEstimation(pr.Processor):
#     def __init__(self, hand_estimator, size=128, draw=True):
#         super(HandPoseEstimation).__init__()
#         self.size = size
#         self.draw = draw
#         self.resize_image = pr.ResizeImage((size, size))
#         self.get_scaling_factor = pe.GetScalingFactor(scaling_factor=4)
#         self.expand_dims = pr.ExpandDims(axis=0)
#         self.hand_estimator = hand_estimator
#         self.draw_skeleton = pe.DrawHandSkeleton()
#         self.wrap = pr.WrapOutput(['image', 'keypoints3D', 'keypoints2D'])

#     def call(self, input_image, flip_input_image=False):
#         image = self.resize_image(input_image)
#         scale = self.get_scaling_factor(input_image, self.size)

#         if flip_input_image:
#             image = flip_left_right(image)

#         image = self.expand_dims(image).astype(np.float32)
#         keypoints3D, keypoints2D = self.hand_estimator.predict(image)[:2]

#         keypoints2D = flip_left_right(keypoints2D)
#         keypoints2D = np.array(keypoints2D*scale, dtype=np.uint)
#         if self.draw:
#             image = self.draw_skeleton(input_image, keypoints2D)
#         return self.wrap(image, keypoints3D, keypoints2D)


# class MANOHandPoseEstimation(HandPoseEstimation):
#     def __init__(self):
#         # detect_hand = DetNet()
#         detect_hand = ModelPipeline()
#         super(MANOHandPoseEstimation, self).__init__(detect_hand)


class HandPoseEstimation(pr.Processor):
    def __init__(self, hand_estimator, size=128, draw=True):
        super(HandPoseEstimation).__init__()
        self.size = size
        self.draw = draw
        self.resize_image = pr.ResizeImage((size, size))
        self.get_scaling_factor = pe.GetScalingFactor(scaling_factor=4)
        self.expand_dims = pr.ExpandDims(axis=0)
        self.hand_estimator = hand_estimator
        self.draw_skeleton = pe.DrawHandSkeleton()
        # self.wrap = pr.WrapOutput(['image', 'keypoints3D', 'keypoints2D'])
        self.wrap = pr.WrapOutput(['joint_angles', 'absolute_joint_angle_quaternions', 'global_pos_joints', 'uv'])

    def call(self, input_image, flip_input_image=False):
        image = self.resize_image(input_image)
        scale = self.get_scaling_factor(input_image, self.size)

        if flip_input_image:
            image = flip_left_right(image)

        image = self.expand_dims(image).astype(np.float32)
        
        global_pos_joints, theta_mpii, uv = self.hand_estimator.process(image)
        print(global_pos_joints, theta_mpii, uv)
        theta_mpii = np.squeeze(theta_mpii)
        print(global_pos_joints.shape, theta_mpii.shape, uv.shape)
        absolute_angle_quaternions = mpii_to_mano(theta_mpii)  # quaternions
        joint_angles = calculate_handstate_joint_angles_from_min_hand_absolute_angles(absolute_angle_quaternions)

        global_pos_joints = mpii_to_mano(global_pos_joints)  # quaternions
        # print(joint_angles, absolute_angle_quaternions, global_pos_joints, uv)
        # return joint_angles, absolute_angle_quaternions, global_pos_joints, uv
        return self.wrap(joint_angles, absolute_angle_quaternions, global_pos_joints, uv)


class MANOHandPoseEstimation(HandPoseEstimation):
    def __init__(self):
        detect_hand = ModelPipeline()
        super(MANOHandPoseEstimation, self).__init__(detect_hand)



class ModelPipeline:
    """
  A wrapper that puts DetNet and IKNet together.
  """

    def __init__(self, left=True):
        # load reference MANO hand pose
        if left:
            data = load_json(HAND_MESH_MODEL_LEFT_PATH_JSON)
        else:
            data = load_json(HAND_MESH_MODEL_RIGHT_PATH_JSON)

        mano_ref_xyz = data['joints']

        # mano_ref_xyz = np.ones(shape=(21,3))

        # convert the kinematic definition to MPII style, and normalize it
        mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
        mpii_ref_xyz -= mpii_ref_xyz[9:10]
        # get bone orientations in the reference pose
        mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
        mpii_ref_delta = mpii_ref_delta * mpii_ref_length

        self.mpii_ref_xyz = mpii_ref_xyz
        self.mpii_ref_delta = mpii_ref_delta

        self.det_model = DetNet()
        # 84 = 21 joint coordinates
        #    + 21 bone orientations
        #    + 21 joint coordinates in reference pose
        #    + 21 bone orientations in reference pose
        
        self.ik_model = ModelIK()
        # self.ik_model = ModelIK(84, IK_MODEL_PATH, 6, 1024)

    def process(self, frame):
        """
        Process a single frame.

        Parameters
        ----------
        frame : np.ndarray, shape [128, 128, 3], dtype np.uint8.
          Frame to be processed.

        Returns
        -------
        np.ndarray, shape [21, 3]
          Joint locations.
        np.ndarray, shape [21, 4]
          Joint rotations.
        """
        xyz, uv = self.det_model.predict(frame)[:2]
        # xyz, uv = self.det_model.process(frame)
        delta, length = xyz_to_delta(xyz, MPIIHandJoints)

        delta *= length
        pack = np.concatenate(
            [xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta], 0
        )
        print(pack.shape)
        pack = np.expand_dims(pack, 0)
        print('****************************')
        print('****************************')
        print(pack.shape)
        print('****************************')
        print('****************************')
        # theta = self.ik_model.process(pack)
        theta = self.ik_model.predict(pack)
        return xyz, theta, uv


    def process_3d_joints(self, xyz):
        delta, length = xyz_to_delta(xyz, MPIIHandJoints)

        delta *= length
        pack = np.concatenate(
            [xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta], 0
        )
        theta = self.ik_model.process(pack)
        return xyz, theta