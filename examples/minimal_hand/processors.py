
from paz.processors import Processor
from paz.backend.image import draw_keypoints_link
from paz.backend.image import draw_keypoints
from joint_config import VISUALISATION_CONFIG

class DrawHandSkeleton(Processor):
    """ Draw human pose skeleton on image.

    # Arguments
        images: Numpy array.
        grouped_joints: Joint locations of all the person model detected
                        in the image. List of numpy array.
        dataset: String.
        check_scores: Boolean. Flag to check score before drawing.

    # Returns
        A numpy array containing pose skeleton.
    """
    def __init__(self, dataset, check_scores=False):
        super(DrawHandSkeleton, self).__init__()
        self.link_orders = VISUALISATION_CONFIG[dataset]['part_orders']
        self.link_colors = VISUALISATION_CONFIG[dataset]['part_color']
        self.link_args = VISUALISATION_CONFIG[dataset]['part_arg']
        self.keypoint_colors = VISUALISATION_CONFIG[dataset]['joint_color']
        self.check_scores = check_scores

    def call(self, image, grouped_joints):
        for one_person_joints in grouped_joints:
            image = draw_keypoints_link(
                image, one_person_joints, self.link_args, self.link_orders,
                self.link_colors, self.check_scores)
            image = draw_keypoints(image, one_person_joints,
                                   self.keypoint_colors, self.check_scores)
        return image
