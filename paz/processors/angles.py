from paz import processors as pr
from paz.backend.angles import change_link_order
from paz.datasets import MANOHandJoints
from paz.backend.keypoints import flip_along_x_axis
from paz.backend.groups import quaternions_to_rotation_matrices
from paz.backend.groups import to_affine_matrices
from paz.backend.keypoints import rotate_keypoints3D
from paz.backend.keypoints import compute_orientation_vector
from paz.backend.angles import calculate_relative_angle
from paz.backend.angles import reorder_relative_angles
from paz.backend.angles import is_hand_open
from paz.datasets import MPIIHandJoints
from paz.datasets.CMU_poanoptic import hand_part_arg


class ChangeLinkOrder(pr.Processor):
    """Map data from one config to another.

    # Arguments
        joints: Array
        config1_labels: input joint configuration
        config2_labels: output joint configuration

    # Returns
        Array: joints maped to the config2_labels
    """
    def __init__(self, config1_labels, config2_labels):
        super(ChangeLinkOrder, self).__init__()
        self.config1_labels = config1_labels
        self.config2_labels = config2_labels

    def call(self, joints):
        mapped_joints = change_link_order(joints, self.config1_labels,
                                          self.config2_labels)
        return mapped_joints


class CalculateRelativeAngles(pr.Processor):
    """Compute the realtive joint rotation for the minimal hand joints and map
       it to the output_config kinematic chain form.

    # Arguments
        absolute_quaternions : Array [num_joints, 4].
        Absolute joint angle rotation for the minimal hand joints in
        quaternion representation [q1, q2, q3, w0].

    # Returns
        relative_angles: Array [num_joints, 3].
        Relative joint rotation of the minimal hand joints in compact
        axis angle representation.
    """
    def __init__(self, right_hand=False, input_config=MANOHandJoints,
                 output_config=MPIIHandJoints):
        super(CalculateRelativeAngles, self).__init__()
        output_labels = output_config.labels
        input_labels = input_config.labels
        links_origin = input_config.links_origin
        self.parents = input_config.parents
        self.children = output_config.children
        if right_hand:
            links_origin = flip_along_x_axis(links_origin)
        self.links_orientation = compute_orientation_vector(
            links_origin, self.parents)
        self.quaternions_to_rotations = pr.SequentialProcessor([
            pr.ChangeLinkOrder(output_labels, input_labels),
            quaternions_to_rotation_matrices])
        self.calculate_relative_angle = pr.SequentialProcessor([
            calculate_relative_angle,
            pr.ChangeLinkOrder(input_labels, output_labels)])

    def call(self, absolute_quaternions):
        absolute_rotation = self.quaternions_to_rotations(absolute_quaternions)
        rotated_links_origin = rotate_keypoints3D(
            absolute_rotation, self.links_orientation)
        rotated_links_origin_transform = to_affine_matrices(
            absolute_rotation, rotated_links_origin)
        relative_angles = self.calculate_relative_angle(
            absolute_rotation, rotated_links_origin_transform, self.parents)
        relative_angles = reorder_relative_angles(
            relative_angles, absolute_rotation[0], self.children)
        return relative_angles


class IsHandOpen(pr.Processor):
    """Check is the hand is open by by using the relative angles of the joint.
    # Arguments
        joint_name_to_arg: Dictionary for the joints
        thresh: Float. Threshold value for theta
        relative_angle: Array

    # Returns
        String: Hand is open or closed.
    """
    def __init__(self, joint_name_to_arg=hand_part_arg, thresh=0.4):
        super(IsHandOpen, self).__init__()
        self.joint_name_to_arg = joint_name_to_arg
        self.thresh = thresh

    def call(self, relative_angles):
        return is_hand_open(relative_angles, self.joint_name_to_arg,
                            self.thresh)
