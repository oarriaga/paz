from paz import processors as pr
from paz.models import IKNet
from paz.datasets import MPIIHandJoints
from paz.datasets import MANOHandJoints
from paz.backend.standard import transform_column_to_negative
from paz.backend.keypoints import get_links_origin


class IKNetHandJointAngles(pr.Processor):
    """Estimate absolute and relative joint angle for the minimal hand joints
       using the 3D keypoint locations.

    # Arguments
        right_hand: Boolean. If 'True', estimate angles for right hand, else
                    estimate angles for left hand.
        keypoints3D: Array [num_joints, 3]. 3D location of keypoints.

    # Returns
        absolute_angles: Array [num_joints, 4]. quaternion repesentation
        relative_angles: Array [num_joints, 3]. axis-angle repesentation
    """
    def __init__(self, right_hand=False, config=MPIIHandJoints):
        super(IKNetHandJointAngles, self).__init__()
        self.calculate_orientation = pr.CalculateOrientationFromCoordinates(
            MPIIHandJoints)
        self.links_origin = config.links_origin
        self.right_hand = right_hand
        if self.right_hand:
            self.links_origin = transform_column_to_negative(self.links_origin)
        self.links_delta = self.calculate_orientation(self.links_origin)
        self.concatenate = pr.Concatenate(0)
        self.compute_absolute_angle = pr.SequentialProcessor(
            [pr.ExpandDims(0), IKNet(), pr.Squeeze(0)])
        mano_links_origin = get_links_origin(MANOHandJoints, right_hand)
        self.compute_relative_angle = pr.SequentialProcessor([
            pr.MapJointConfig(MPIIHandJoints, MANOHandJoints),
            pr.ComputeRelativeAngles(mano_links_origin, right_hand)])
        self.wrap = pr.WrapOutput(['absolute_angles', 'relative_angles'])

    def call(self, keypoints3D):
        delta = self.calculate_orientation(keypoints3D)
        pack = self.concatenate(
            [keypoints3D, delta, self.links_origin, self.links_delta])
        absolute_angles = self.compute_absolute_angle(pack)
        relative_angles = self.compute_relative_angle(absolute_angles)
        return self.wrap(absolute_angles, relative_angles)
