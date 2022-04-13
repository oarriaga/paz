import numpy as np
import processors as pe
from paz import processors as pr
from paz.backend.image import flip_left_right
from models.detnet import DetNet
from models.iknet import ModelIK


from backend import relative_angle_quaternions
from joint_config import MANO_REF_JOINTS, IK_UNIT_LENGTH
from joint_config import MANOHandJoints, MPIIHandJoints


class HandPoseEstimation(pr.Processor):
    def __init__(self, hand_estimator, size=128, draw=True):
        super(HandPoseEstimation).__init__()
        self.size = size
        self.draw = draw
        self.resize_image = pr.ResizeImage((size, size))
        self.get_scaling_factor = pe.GetScalingFactor(scaling_factor=4)
        self.expand_dims = pr.ExpandDims(axis=0)
        self.hand_pose_estimator = hand_estimator
        self.draw_skeleton = pe.DrawHandSkeleton()
        self.mpii_to_mano = pe.MapJointConfig(MANOHandJoints, MPIIHandJoints)
        self.wrap = pr.WrapOutput(['relative_joint_angles',
                                   'absolute_joint_angle',
                                   'keypoints3D', 'keypoints2D'])

    def call(self, input_image, flip_input_image=False):
        image = self.resize_image(input_image)
        scale = self.get_scaling_factor(input_image, self.size)

        if flip_input_image:
            image = flip_left_right(image)

        image = self.expand_dims(image).astype(np.float32)

        keypoints3D, theta_mpii, keypoints2D = self.hand_pose_estimator(image)
        theta_mpii = np.squeeze(theta_mpii)
        absolute_joint_angle = self.mpii_to_mano(theta_mpii) 
        relative_joint_angles = relative_angle_quaternions(absolute_joint_angle)


        # keypoints2D = flip_left_right(keypoints2D)
        # keypoints2D = np.array(keypoints2D*scale, dtype=np.uint)
        # if self.draw:
        #     image = self.draw_skeleton(input_image, keypoints2D)

        keypoints3D = self.mpii_to_mano(keypoints3D)  # quaternions
        return self.wrap(relative_joint_angles, absolute_joint_angle,
                         keypoints3D, keypoints2D)


class MANOHandPoseEstimation(HandPoseEstimation):
    def __init__(self):
        detect_hand = HandPoseEstimatorModel()
        super(MANOHandPoseEstimation, self).__init__(detect_hand)


class HandPoseEstimatorModel(pr.Processor):
    def __init__(self, left=True):
        super(HandPoseEstimatorModel, self).__init__()
        self.mano_to_mpii = pe.MapJointConfig(MPIIHandJoints, MANOHandJoints)
        self.calculate_orientation = pe.CalculateOrientationFromCoordinates(
            MPIIHandJoints)

        if left:
            mano_ref_xyz = MANO_REF_JOINTS
        else:
            pass
            # -1 * first coloumn

        mpii_ref_xyz = self.mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
        mpii_ref_xyz = mpii_ref_xyz - mpii_ref_xyz[9:10]
        mpii_ref_delta = self.calculate_orientation(mpii_ref_xyz)

        self.mpii_ref_xyz = mpii_ref_xyz
        self.mpii_ref_delta = mpii_ref_delta

        self.det_model = DetNet()
        self.ik_model = ModelIK()

    def call(self, image):
        xyz, uv = self.det_model.predict(image)[:2]

        delta = self.calculate_orientation(xyz)

        pack = np.concatenate(
            [xyz, delta, self.mpii_ref_xyz, self.mpii_ref_delta], 0)
        pack = np.expand_dims(pack, 0)
        theta = self.ik_model.predict(pack)
        return xyz, theta, uv
