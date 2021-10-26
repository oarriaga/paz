import numpy as np

from paz.abstract import Processor
from paz.backend.image.tensorflow_image import resize
from ..backend.keypoints import create_score_maps, extract_2D_keypoints
from ..backend.keypoints import crop_image_from_coordinates, detect_keypoints
from ..backend.keypoints import crop_image_using_mask, extract_hand_segment
from ..backend.keypoints import extract_bounding_box, find_max_location
from ..backend.keypoints import extract_dominant_hand_visibility
from ..backend.keypoints import extract_dominant_keypoints2D, flip_right_hand
from ..backend.keypoints import get_hand_side_and_keypooints
from ..backend.keypoints import normalize_keypoints


class ExtractHandmask(Processor):
    """Extract Hand mask."""

    def __init__(self):
        super(ExtractHandmask, self).__init__()

    def call(self, segmentation_label):
        return extract_hand_segment(segmentation_label=segmentation_label)


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
        crop_size_best = crop_size_best.astype(dtype=np.float64)
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


class FindMaxLocation(Processor):
    """ Find the brightest point in the score map, which is represented as a
    keypoint"""
    def __init__(self):
        super(FindMaxLocation, self).__init__()

    def call(self, scoremaps):
        keypoints_2D = find_max_location(scoremaps)
        return keypoints_2D