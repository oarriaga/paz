from paz import processors as pr
from paz.models import IKNet
from paz.datasets import MPIIHandJoints
from paz.backend.keypoints import flip_along_x_axis


class IKNetHandJointAngles(pr.Processor):
    """Estimate absolute and relative joint angle for the minimal hand joints
       using the 3D keypoint locations.

    # Arguments
        links_origin: Array. Reference pose of the minimal hand joints.
        parent: List. Parents of the keypoints from kinematic chain
        right_hand: Boolean. If 'True', estimate angles for right hand, else
                    estimate angles for left hand.
        keypoints3D: Array [num_joints, 3]. 3D location of keypoints.

    # Returns
        absolute_angles: Array [num_joints, 4]. quaternion repesentation
        relative_angles: Array [num_joints, 3]. axis-angle repesentation
    """
    def __init__(self, links_origin=MPIIHandJoints.links_origin,
                 parents=MPIIHandJoints.parents, right_hand=False):
        super(IKNetHandJointAngles, self).__init__()
        self.calculate_orientation = pr.ComputeOrientationVector(parents)
        self.links_origin = links_origin
        self.right_hand = right_hand
        if self.right_hand:
            self.links_origin = flip_along_x_axis(self.links_origin)
        self.links_delta = self.calculate_orientation(self.links_origin)
        self.concatenate = pr.Concatenate(0)
        self.compute_absolute_angles = pr.SequentialProcessor(
            [pr.ExpandDims(0), IKNet(), pr.Squeeze(0)])
        self.compute_relative_angles = pr.CalculateRelativeAngles()
        self.wrap = pr.WrapOutput(['absolute_angles', 'relative_angles'])

    def call(self, keypoints3D):
        delta = self.calculate_orientation(keypoints3D)
        pack = self.concatenate(
            [keypoints3D, delta, self.links_origin, self.links_delta])
        absolute_angles = self.compute_absolute_angles(pack)
        relative_angles = self.compute_relative_angles(absolute_angles)
        return self.wrap(absolute_angles, relative_angles)

