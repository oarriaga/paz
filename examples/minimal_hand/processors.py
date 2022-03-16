
from paz.processors import Processor
from paz.backend.image import draw_keypoints_link
from paz.backend.image import draw_keypoints
from joint_config import VISUALISATION_CONFIG
from wrappers import ModelPipeline
from backend import get_scaling_factor


class DetectHandKeypoints(Processor):
    def __init__(self):
        super(DetectHandKeypoints, self).__init__()
        self.model = ModelPipeline(left=True)

    def call(self, image):
        keypoints_3D, theta_mpii, keypoints_2D = self.model.process(image)
        return keypoints_3D, keypoints_2D


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

    def call(self, image, keypoints):
        image = draw_keypoints_link(
            image, keypoints, self.link_args, self.link_orders,
            self.link_colors, self.check_scores)
        image = draw_keypoints(image, keypoints,
                               self.keypoint_colors, self.check_scores)
        return image
