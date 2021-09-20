import numpy as np

from backend import create_score_maps, extract_2D_keypoints
from backend import crop_image_from_coordinates, rotation_from_axis_angles
from backend import detect_keypoints, wrap_dictionary, merge_dictionaries
from backend import extract_dominant_hand_visibility, extract_bounding_box
from backend import extract_dominant_keypoints2D, crop_image_using_mask
from backend import extract_hand_segment, keypoints_to_wrist_coordinates
from backend import get_bone_connections_and_colors, find_max_location
from backend import get_canonical_transformations, flip_right_hand
from backend import normalize_keypoints, transform_to_relative_frames
from backend import transform_cropped_keypoints
from backend import transform_visibility_mask, get_hand_side_and_keypooints
from paz.abstract import Processor
from paz.backend.image.draw import lincolor
from paz.backend.image.tensorflow_image import resize


class ExtractHandmask(Processor):
    """Extract Hand mask."""
    def __init__(self):
        super(ExtractHandmask, self).__init__()

    def call(self, segmentation_label):
        return extract_hand_segment(segmentation_label=segmentation_label)


class KeypointsWristFrame(Processor):
    """Translate to Wrist Coordinates.
    """

    def __init__(self):
        super(KeypointsWristFrame, self).__init__()

    def call(self, keypoints):
        return keypoints_to_wrist_coordinates(keypoints=keypoints)


class TransformVisibilityMask(Processor):
    """Extract Visibility Mask.
    """

    def __init__(self, use_wrist_coordinates):
        super(TransformVisibilityMask, self).__init__()
        self.use_wrist_coordinates = use_wrist_coordinates

    def call(self, visibility_mask):
        return transform_visibility_mask(visibility_mask)


class ExtractHandSide(Processor):
    """Extract Hand Side.
    """

    def __init__(self):
        super(ExtractHandSide, self).__init__()

    def call(self, hand_parts_mask, keypoints3D):
        return get_hand_side_and_keypooints(hand_parts_mask, keypoints3D)


class NormalizeKeypoints(Processor):
    """Normalize KeyPoints.
        """

    def __init__(self):
        super(NormalizeKeypoints, self).__init__()

    def call(self, keypoints3D):
        return normalize_keypoints(keypoints3D)


class TransformtoRelativeFrame(Processor):
    """Transform to Relative Frame."""

    def __init__(self):
        super(TransformtoRelativeFrame, self).__init__()

    def call(self, keypoints3D):
        return np.squeeze(transform_to_relative_frames(keypoints3D))


class GetCanonicalTransformation(Processor):
    """Extract Canonical Transformation matrix.
        """

    def __init__(self):
        super(GetCanonicalTransformation, self).__init__()

    def call(self, keypoints3D):
        return get_canonical_transformations(keypoints3D)


class FlipRightHand(Processor):
    """Flip Right Hand to Left Hand coordinates.
        """

    def __init__(self, flip_to_left=True):
        super(FlipRightHand, self).__init__()
        self.flip_to_left = flip_to_left

    def call(self, keypoints3D):
        return flip_right_hand(keypoints3D, self.flip_to_left)


class ExtractDominantHandVisibility(Processor):
    """Extract Dominant hand Visibility.
        """

    def __init__(self):
        super(ExtractDominantHandVisibility, self).__init__()

    def call(self, keypoint_visibility, dominant_hand):
        return extract_dominant_hand_visibility(keypoint_visibility,
                                                dominant_hand)


class ExtractDominantKeypoint(Processor):
    """Extract Dominant hand Keypoints.
        """

    def __init__(self):
        super(ExtractDominantKeypoint, self).__init__()

    def call(self, keypoint_visibility, dominant_hand):
        return extract_dominant_keypoints2D(keypoint_visibility,
                                            dominant_hand)


class CropImageFromMask(Processor):
    """Crop Image from Mask.
            """

    def __init__(self, image_size=(320, 320, 3), crop_size=256):
        super(CropImageFromMask, self).__init__()
        self.image_size = image_size
        self.crop_size = crop_size

    def call(self, keypoints, keypoint_visibility, image, camera_matrix):
        return crop_image_using_mask(keypoints, keypoint_visibility, image,
                                     self.image_size, self.crop_size,
                                     camera_matrix)


class CreateScoremaps(Processor):
    """Create Gaussian Score maps representing 2D Keypoints.
            """

    def __init__(self, image_size, crop_size, variance):
        super(CreateScoremaps, self).__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.variance = variance

    def call(self, keypoints2D, keypoints_vis21):
        return create_score_maps(keypoints2D, keypoints_vis21, self.image_size,
                                 self.crop_size, self.variance)


class Extract2DKeypoints(Processor):
    """ Extract the keyppoints based on the visibility of the hand"""
    def __init__(self):
        super(Extract2DKeypoints, self).__init__()

    def call(self, keypoint_visibility):
        return extract_2D_keypoints(keypoint_visibility)


class MatrixInverse(Processor):
    """ Perform Pseudo Inverse of the matrix"""
    def __init__(self):
        super(MatrixInverse, self).__init__()

    def call(self, matrix):
        return np.linalg.pinv(matrix)


class ExtractBoundingbox(Processor):
    """ Extract bounding box when provided with a binary mask"""
    def __init__(self):
        super(ExtractBoundingbox, self).__init__()

    def call(self, binary_hand_mask):
        return extract_bounding_box(binary_hand_mask)


class AdjustCropSize(Processor):
    """ Adjust the crop size with a buffer of scale 0.25 added"""
    def __init__(self, crop_size=256):
        super(AdjustCropSize, self).__init__()
        self.crop_size = crop_size

    def call(self, crop_size_best):
        crop_size_best *= 1.25
        scaled_crop = np.maximum(self.crop_size / crop_size_best, 0.25)
        scaled_crop = np.minimum(scaled_crop, 5.0)
        return scaled_crop


class CropImage(Processor):
    """ Crop the input image provided the location, output image size and the
    scaling of the output image"""
    def __init__(self, crop_size=256):
        super(CropImage, self).__init__()
        self.crop_size = crop_size

    def call(self, image, crop_location, scale):
        return crop_image_from_coordinates(image, crop_location, self.crop_size,
                                           scale)


class RotationMatrixfromAxisAngles(Processor):
    """ Get Rotation matrix from the axis angles"""
    def __init__(self):
        super(RotationMatrixfromAxisAngles, self).__init__()

    def call(self, rotation_angles):
        return rotation_from_axis_angles(rotation_angles)


class CanonicaltoRelativeFrame(Processor):
    """ Transform the keypoints from Canonical coordinates to chosen relative (
    wrist or palm) coordinates """
    def __init__(self, num_keypoints=21):
        super(CanonicaltoRelativeFrame, self).__init__()
        self.num_keypoints = num_keypoints

    def call(self, canonical_coordinates, rotation_matrix, hand_side):
        cond_right = np.equal(np.argmax(hand_side, 1), 1)
        cond_right_all = np.tile(np.reshape(cond_right, [-1, 1, 1]),
                                 [1, self.num_keypoints, 3])

        coord_xyz_can_flip = flip_right_hand(canonical_coordinates,
                                             cond_right_all)
        # rotate view back
        coord_xyz_rel_normed = np.matmul(coord_xyz_can_flip, rotation_matrix)
        return coord_xyz_rel_normed


class ExtractKeypoints(Processor):
    """ Extract keypoints when provided with a predicted scoremap"""
    def __init__(self):
        super(ExtractKeypoints, self).__init__()

    def call(self, keypoint_scoremaps):
        return detect_keypoints(keypoint_scoremaps)


class Resize_image(Processor):
    """ Resize images. Done with tensorflow"""
    def __init__(self, size=[256, 256]):
        super(Resize_image, self).__init__()
        self.size = size

    def call(self, image):
        return resize(image, self.size)


class WraptoDictionary(Processor):
    """ Wrap the input values to a dictionary with already provided key
    values """
    def __init__(self, keys):
        super(WraptoDictionary, self).__init__()
        if not isinstance(keys, list):
            keys = list(keys)
        self.keys = keys

    def call(self, values):
        if not isinstance(values, list):
            values = list(values)
        return wrap_dictionary(self.keys, values)


class MergeDictionaries(Processor):
    """ Merge two dictionaries into one"""
    def __init__(self):
        super(MergeDictionaries, self).__init__()

    def call(self, dicts):
        return merge_dictionaries(dicts)


class GetBoneColorEncoding(Processor):
    """ Map bone segmentation with respective colors to visualize"""
    def __init__(self):
        super(GetBoneColorEncoding, self).__init__()

    def call(self, num_keypoints=21):
        colors = lincolor(num_colors=num_keypoints)
        return get_bone_connections_and_colors(colors=colors)


class FindMaxLocation(Processor):
    """ Find the brightest point in the score map, which is represented as a
    keypoint"""
    def __init__(self):
        super(FindMaxLocation, self).__init__()

    def call(self, scoremaps):
        keypoints_2D = find_max_location(scoremaps)
        return keypoints_2D


class TransformKeypoints(Processor):
    """ Transform the keypoint from cropped image frame to original image
    frame"""
    def __init__(self):
        super(TransformKeypoints, self).__init__()

    def call(self, cropped_keypoints, centers, scale, crop_size):
        keypoints_2D = transform_cropped_keypoints(cropped_keypoints, centers,
                                                   scale, crop_size)
        return keypoints_2D