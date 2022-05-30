from ..abstract import Processor
from ..backend.angles import compute_relative_angles
from ..backend.standard import map_joint_config


class MapJointConfig(Processor):
    def __init__(self, joint_config1, joint_config2):
        super(MapJointConfig, self).__init__()
        self.joint_config1 = joint_config1
        self.joint_config2 = joint_config2

    def call(self, joints):
        mapped_joints = map_joint_config(joints, self.joint_config1,
                                         self.joint_config2)
        return mapped_joints


class ComputeRelativeAngles(Processor):
    def __init__(self, links_origin, right_hand):
        super(ComputeRelativeAngles, self).__init__()
        self.links_origin = links_origin
        self.right_hand = right_hand

    def call(self, absolute_angles):
        relative_angles = compute_relative_angles(
            absolute_angles, self.links_origin, self.right_hand)
        return relative_angles
