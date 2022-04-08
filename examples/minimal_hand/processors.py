
from paz.processors import Processor
from paz.backend.image import draw_keypoints_link
from paz.backend.image import draw_keypoints
from joint_config import VISUALISATION_CONFIG
from backend import get_scaling_factor


class GetScalingFactor(Processor):
    def __init__(self, scaling_factor):
        super(GetScalingFactor, self).__init__()
        self.scaling_factor = scaling_factor

    def call(self, image, size):
        return get_scaling_factor(image, size, self.scaling_factor)


class DrawHandSkeleton(Processor):
    """ Draw human pose skeleton on image.

    # Arguments
        images: Numpy array.
        grouped_joints: Joint locations of all the person model detected
                        in the image. List of numpy array.
        check_scores: Boolean. Flag to check score before drawing.

    # Returns
        A numpy array containing pose skeleton.
    """
    def __init__(self, check_scores=False):
        super(DrawHandSkeleton, self).__init__()
        self.link_orders = VISUALISATION_CONFIG['part_orders']
        self.link_colors = VISUALISATION_CONFIG['part_color']
        self.link_args = VISUALISATION_CONFIG['part_arg']
        self.keypoint_colors = VISUALISATION_CONFIG['joint_color']
        self.check_scores = check_scores

    def call(self, image, keypoints, link_width=2, keypoint_radius=4):
        image = draw_keypoints_link(
            image, keypoints, self.link_args, self.link_orders,
            self.link_colors, self.check_scores, link_width)
        image = draw_keypoints(image, keypoints, self.keypoint_colors,
                               self.check_scores, keypoint_radius)
        return image
