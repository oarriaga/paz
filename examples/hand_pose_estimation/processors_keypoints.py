import numpy as np

from backend_keypoints import create_score_maps, extract_2D_keypoints
from backend_keypoints import crop_image_from_coordinates, extract_keypoints
from backend_keypoints import crop_image_from_mask, extract_hand_segment
from backend_keypoints import extract_bounding_box, find_max_location
from backend_keypoints import extract_dominant_hand_visibility
from backend_keypoints import extract_dominant_keypoints2D
from backend_keypoints import flip_right_to_left_hand
from backend_keypoints import get_hand_side_and_keypooints
from backend_keypoints import normalize_keypoints

from paz.abstract import Processor


class ExtractHandmask(Processor):
    """Extract Hand mask from the segmentation label provided. The pixels
    with value greater than 1 belongs to hands
    """

    def __init__(self):
        super(ExtractHandmask, self).__init__()

    def call(self, segmentation_label):
        return extract_hand_segment(segmentation_label=segmentation_label)


class ExtractHandsideandKeypoints(Processor):
    """Extract Hand Side by counting the number of pixels belonging to each
    hand.
    """

    def __init__(self):
        super(ExtractHandsideandKeypoints, self).__init__()

    def call(self, hand_parts_mask, keypoints3D):
        return get_hand_side_and_keypooints(hand_parts_mask, keypoints3D)


class NormalizeKeypoints(Processor):
    """Normalize KeyPoints.
    """

    def __init__(self):
        super(NormalizeKeypoints, self).__init__()

    def call(self, keypoints3D):
        return normalize_keypoints(keypoints3D)


class FlipRightHandToLeftHand(Processor):
    """Flip Right hand keypoints to Left hand keypoints.
    """

    def __init__(self, flip_to_left=True):
        super(FlipRightHandToLeftHand, self).__init__()
        self.flip_to_left = flip_to_left

    def call(self, keypoints3D):
        return flip_right_to_left_hand(keypoints3D, self.flip_to_left)


class ExtractDominantHandVisibility(Processor):
    """Extract hand Visibility of Left or Right hand based on the
    dominant_hand flag.
        """

    def __init__(self):
        super(ExtractDominantHandVisibility, self).__init__()

    def call(self, keypoint_visibility, dominant_hand):
        return extract_dominant_hand_visibility(keypoint_visibility,
                                                dominant_hand)


class ExtractDominantKeypoints2D(Processor):
    """Extract hand keypoints of Left or Right hand based on the
    dominant_hand flag.
        """

    def __init__(self):
        super(ExtractDominantKeypoints2D, self).__init__()

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
        return crop_image_from_mask(keypoints, keypoint_visibility, image,
                                    self.image_size, self.crop_size,
                                    camera_matrix)


class CreateScoremaps(Processor):
    """Create Gaussian Score maps representing 2D Keypoints.
       image_size: Size of the input image
       crop_size: Cropped Image size
       variance: variance of the gaussian scoremap to be generated
            """

    def __init__(self, image_size, crop_size, variance):
        super(CreateScoremaps, self).__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.variance = variance

    def call(self, keypoints2D, keypoints_visibility):
        return create_score_maps(keypoints2D, keypoints_visibility,
                                 self.image_size, self.crop_size, self.variance)


class Extract2DKeypoints(Processor):
    """ Extract the keyppoints based on the visibility of the hand"""

    def __init__(self):
        super(Extract2DKeypoints, self).__init__()

    def call(self, keypoint_visibility):
        return extract_2D_keypoints(keypoint_visibility)


class ExtractBoundingbox(Processor):
    """ Extract bounding box from a binary mask"""

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
        crop_size_best = crop_size_best * 1.25
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
        return extract_keypoints(keypoint_scoremaps)


class FindMaxLocation(Processor):
    """ Find the brightest point in the score map, which is represented as a
    keypoint"""

    def __init__(self):
        super(FindMaxLocation, self).__init__()

    def call(self, scoremaps):
        keypoints_2D = find_max_location(scoremaps)
        return keypoints_2D
