from minimal_hand_estimator import MinimalHandEstimator
from paz.abstract import Processor

from paz import processors as pr
from paz.backend.image import flip_left_right


class MANOHandPoseEstimation(Processor):
    """MANO Shape Estimation pipeline using a pre-trained ``DetNet`` 3D hand keypoints estimator
        and a pre-trained ``IKNet`` estimation model.

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera`` with
                camera intrinsics.
            radius: Int. radius of keypoint to be drawn.

        # Returns
            A function that takes an RGB image and outputs the following
            inferences as keys of a dictionary:
                ``image``, ``joint_locations``, ``joint_rotations``.

            formats:
                joint_locations: shape(16,3) / root relative xyz locations of 16 hand joints
                joint_rotations: shape(16,3) / joint relative rotations of 16 hand joints in axis angle repr.
        """

    def __init__(self):
        super().__init__()
        self.hand_estimator = MinimalHandEstimator()
        self.wrap = pr.WrapOutput(['joint_angles', 'absolute_joint_angle_quaternions', 'image', 'input_image','global_pos_joints'])

    def call(self, input_image, flip_input_image=False, load_by_25D_keypoints_from_mediapipe=False):


        if load_by_25D_keypoints_from_mediapipe:
            global_pos_joints, mano_joints_xyz, annotated_image = self.media_pipe_hand_keypoint_estimator.predict(
                image=input_image
            )
            mpii_3d_joints = self.minimal_hand_estimator.scale_media_pipe_to_mppi_ref_size(mano_joints_xyz)
            mpii_3d_joints = mpii_3d_joints / self.UNIT_LENGTH

            joint_angles, absolute_joint_angle_quaternions = self.minimal_hand_estimator.predict_from_3d_joints(
                mpii_3d_joints)
        else:

            size = 128
            resize_img = pr.SequentialProcessor([pr.ResizeImage((size, size))])
            image = resize_img(input_image)

            if flip_input_image:
                image = flip_left_right(image)

            joint_angles, absolute_joint_angle_quaternions, global_pos_joints = self.hand_estimator.predict(image)

        print(joint_angles)

        return self.wrap(joint_angles, absolute_joint_angle_quaternions, image, input_image,global_pos_joints)
