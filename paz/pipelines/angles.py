from paz import processors as pr
from paz.models import IKNet
from paz.datasets import MPIIHandJoints
from paz.backend.angles import compute_relative_angle
from paz.backend.standard import transform_column_to_negative


class IKNetHandJointAngles(pr.Processor):
    def __init__(self, right_hand=False, config=MPIIHandJoints):
        super(IKNetHandJointAngles, self).__init__()
        self.calculate_orientation = pr.CalculateOrientationFromCoordinates(
            MPIIHandJoints)
        self.ref_xyz = config.ref_joints
        self.right_hand = right_hand
        if right_hand:
            self.ref_xyz = transform_column_to_negative(self.ref_xyz)
        self.ref_delta = self.calculate_orientation(self.ref_xyz)
        self.concatenate = pr.Concatenate(0)
        self.compute_absolute_angle = pr.SequentialProcessor(
            [pr.ExpandDims(0), IKNet(), pr.Squeeze(0)])
        self.wrap = pr.WrapOutput(['absolute_angles', 'relative_angles'])

    def call(self, keypoints3D):
        delta = self.calculate_orientation(keypoints3D)
        pack = self.concatenate(
            [keypoints3D, delta, self.ref_xyz, self.ref_delta])
        absolute_angle = self.compute_absolute_angle(pack)
        relative_angles = compute_relative_angle(
            absolute_angle, self.right_hand)
        return self.wrap(absolute_angle, relative_angles)