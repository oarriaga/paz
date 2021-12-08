import numpy as np

from backend_SE3 import rotation_from_axis_angles
from backend_keypoints import canonical_to_relative_coordinates
from backend_keypoints import canonical_transformations_on_keypoints
from backend_keypoints import keypoint_to_root_frame
from backend_keypoints import keypoints_to_palm_coordinates
from backend_keypoints import transform_cropped_keypoints
from backend_keypoints import transform_visibility_mask
from paz.abstract import Processor


class TransformKeypoints(Processor):
    """ Transform the keypoint from cropped image frame to original image
    frame"""

    def __init__(self):
        super(TransformKeypoints, self).__init__()

    def call(self, cropped_keypoints, centers, scale, crop_size):
        keypoints_2D = transform_cropped_keypoints(cropped_keypoints, centers,
                                                   scale, crop_size)
        return keypoints_2D


class KeypointstoPalmFrame(Processor):
    """Translate to Wrist Coordinates.
    """

    def __init__(self):
        super(KeypointstoPalmFrame, self).__init__()

    def call(self, keypoints):
        return keypoints_to_palm_coordinates(keypoints=keypoints)


class TransformVisibilityMask(Processor):
    """Tranform Visibility Mask to palm coordinates.
    """

    def __init__(self):
        super(TransformVisibilityMask, self).__init__()

    def call(self, visibility_mask):
        return transform_visibility_mask(visibility_mask)


class TransformtoRelativeFrame(Processor):
    """Transform to Relative Frame."""

    def __init__(self):
        super(TransformtoRelativeFrame, self).__init__()

    def call(self, keypoints3D):
        return keypoint_to_root_frame(keypoints3D)


class GetCanonicalTransformation(Processor):
    """Extract Canonical Transformation matrix. To transform keypoints to
    palm frame inorder to make them rotationally invariant
    """

    def __init__(self):
        super(GetCanonicalTransformation, self).__init__()

    def call(self, keypoints3D):
        return canonical_transformations_on_keypoints(keypoints3D)


class CalculatePseudoInverse(Processor):
    """ Perform Pseudo Inverse of the matrix"""

    def __init__(self):
        super(CalculatePseudoInverse, self).__init__()

    def call(self, matrix):
        return np.linalg.pinv(matrix)


class RotationMatrixfromAxisAngles(Processor):
    """ Get Rotation matrix from the axis angles"""

    def __init__(self):
        super(RotationMatrixfromAxisAngles, self).__init__()

    def call(self, rotation_angles):
        return rotation_from_axis_angles(rotation_angles)


class CanonicaltoRelativeFrame(Processor):
    """ Transform the keypoints from Canonical coordinates to chosen relative (
    wrist or palm) coordinates. To make keypoints rotationally invariant """

    def __init__(self, num_keypoints=21):
        super(CanonicaltoRelativeFrame, self).__init__()
        self.num_keypoints = num_keypoints

    def call(self, canonical_coordinates, rotation_matrix, hand_side):
        canonical_coordinates = canonical_coordinates.reshape((21, 3))
        keypoints = canonical_to_relative_coordinates(
            self.num_keypoints, canonical_coordinates, rotation_matrix,
            hand_side)
        return keypoints
