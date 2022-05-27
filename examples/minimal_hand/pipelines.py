import processors as pe
from paz import processors as pr
from paz.models import DetNet
from paz.models import IKNet
from paz.backend.image import flip_left_right
from paz.datasets import MPII_REF_JOINTS
from paz.datasets import MPIIHandJoints
from backend import compute_relative_angle
from backend import transform_column_to_negative
from backend import flip_keypoints_wrt_image


class MinimalHandPoseEstimator(pr.Processor):
    def __init__(self, draw=True, right_hand=False):
        super(MinimalHandPoseEstimator, self).__init__()
        self.keypoints_estimator = DetNetHandKeypoint(draw=draw,
                                                      right_hand=right_hand)
        self.angle_estimator = IKNetHandJointAngles(right_hand)
        self.wrap = pr.WrapOutput(['image', 'keypoints3D', 'keypoints2D',
                                   'absolute_angles', 'relative_angles'])

    def call(self, image):
        keypoints = self.keypoints_estimator(image)
        angles = self.angle_estimator(keypoints['keypoints3D'])
        print(keypoints, angles)
        return self.wrap(keypoints['image'], keypoints['keypoints3D'],
                         keypoints['keypoints2D'], angles['absolute_angles'],
                         angles['relative_angles'])


class DetNetHandKeypoint(pr.Processor):
    def __init__(self, shape=(128, 128), draw=True, right_hand=False):
        super(DetNetHandKeypoint).__init__()
        self.draw = draw
        self.right_hand = right_hand
        self.preprocess = pr.SequentialProcessor(
            [pr.ResizeImage(shape), pr.ExpandDims(axis=0)])
        self.hand_estimator = DetNet()
        self.scale_keypoints = pr.ScaleKeypoints(scale=4, shape=shape)
        self.draw_skeleton = pr.DrawHandSkeleton()
        self.wrap = pr.WrapOutput(['image', 'keypoints3D', 'keypoints2D'])

    def call(self, input_image):
        image = self.preprocess(input_image)
        if self.right_hand:
            image = flip_left_right(image)
        keypoints3D, keypoints2D = self.hand_estimator.predict(image)
        keypoints2D = flip_left_right(keypoints2D)
        if self.right_hand:
            keypoints2D = flip_keypoints_wrt_image(keypoints2D)
        keypoints2D = self.scale_keypoints(keypoints2D, input_image)
        if self.draw:
            image = self.draw_skeleton(input_image, keypoints2D)
        return self.wrap(image, keypoints3D, keypoints2D)


class IKNetHandJointAngles(pr.Processor):
    def __init__(self, right_hand=False, config_xyz=MPII_REF_JOINTS):
        super(IKNetHandJointAngles, self).__init__()
        self.calculate_orientation = pe.CalculateOrientationFromCoordinates(
            MPIIHandJoints)
        self.ref_xyz = config_xyz
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
