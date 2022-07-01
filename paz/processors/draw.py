import numpy as np

from ..abstract import Processor
from ..backend.image import lincolor
from ..backend.image import draw_rectangle
from ..backend.image import put_text
from ..backend.image import draw_keypoint
from ..backend.image import draw_cube
from ..backend.image import GREEN
from ..backend.image import draw_random_polygon
from ..backend.image import draw_keypoints_link
from ..backend.image import draw_keypoints
from ..backend.image import draw_RGB_mask
from ..backend.image import draw_RGB_masks
from ..backend.keypoints import project_points3D
from ..backend.keypoints import build_cube_points3D
from ..backend.groups import quaternion_to_rotation_matrix
from ..backend.keypoints import project_to_image
from ..datasets import HUMAN_JOINT_CONFIG
from ..datasets import MINIMAL_HAND_CONFIG


class DrawBoxes2D(Processor):
    """Draws bounding boxes from Boxes2D messages.

    # Arguments
        class_names: List of strings.
        colors: List of lists containing the color values
        weighted: Boolean. If ``True`` the colors are weighted with the
            score of the bounding box.
        scale: Float. Scale of drawn text.
    """
    def __init__(self, class_names=None, colors=None,
                 weighted=False, scale=0.7, with_score=True):
        self.class_names = class_names
        self.colors = colors
        self.weighted = weighted
        self.with_score = with_score
        self.scale = scale

        if (self.class_names is not None and
                not isinstance(self.class_names, list)):
            raise TypeError("Class name should be of type 'List of strings'")

        if (self.colors is not None and
                not all(isinstance(color, list) for color in self.colors)):
            raise TypeError("Colors should be of type 'List of lists'")

        if self.colors is None:
            self.colors = lincolor(len(self.class_names))

        if self.class_names is not None:
            self.class_to_color = dict(zip(self.class_names, self.colors))
        else:
            self.class_to_color = {None: self.colors, '': self.colors}
        super(DrawBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            class_name = box2D.class_name
            color = self.class_to_color[class_name]
            if self.weighted:
                color = [int(channel * box2D.score) for channel in color]
            if self.with_score:
                text = '{:0.2f}, {}'.format(box2D.score, class_name)
            if not self.with_score:
                text = '{}'.format(class_name)
            put_text(image, text, (x_min, y_min - 10), self.scale, color, 1)
            draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        return image


class DrawKeypoints2D(Processor):
    """Draws keypoints into image.

    # Arguments
        num_keypoints: Int. Used initialize colors for each keypoint
        radius: Float. Approximate radius of the circle in pixel coordinates.
    """
    def __init__(self, num_keypoints, radius=3, normalized=False):
        super(DrawKeypoints2D, self).__init__()
        self.colors = lincolor(num_keypoints, normalized=normalized)
        self.radius = radius

    def call(self, image, keypoints):
        for keypoint_arg, keypoint in enumerate(keypoints):
            color = self.colors[keypoint_arg]
            draw_keypoint(image, keypoint.astype('int'), color, self.radius)
        return image


class DrawBoxes3D(Processor):
    def __init__(self, camera, class_to_dimensions,
                 color=GREEN, thickness=5, radius=2):
        """Draw boxes 3D of multiple objects

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera''.
            class_to_dimensions: Dictionary that has as keys the
                class names and as value a list [model_height, model_width]
            thickness: Int. Thickness of 3D box
        """
        super(DrawBoxes3D, self).__init__()
        self.camera = camera
        self.class_to_dimensions = class_to_dimensions
        self.class_to_points = self._build_class_to_points(class_to_dimensions)
        self.color = color
        self.radius = radius
        self.thickness = thickness

    def _build_class_to_points(self, class_to_dimensions):
        class_to_points = {}
        for class_name, dimensions in self.class_to_dimensions.items():
            width, height, depth = dimensions
            points = build_cube_points3D(width, height, depth)
            class_to_points[class_name] = points
        return class_to_points

    def call(self, image, pose6D):
        points3D = self.class_to_points[pose6D.class_name]
        points2D = project_points3D(points3D, pose6D, self.camera)
        points2D = points2D.astype(np.int32)
        draw_cube(image, points2D, self.color, self.thickness, self.radius)
        return image


class DrawRandomPolygon(Processor):
    """ Adds occlusion to image

    # Arguments
        max_radius_scale: Maximum radius in scale with respect to image i.e.
                each vertex radius from the polygon is sampled
                from ``[0, max_radius_scale]``. This radius is later
                multiplied by the image dimensions.
    """
    def __init__(self, max_radius_scale=.5):
        super(DrawRandomPolygon, self).__init__()
        self.max_radius_scale = max_radius_scale

    def call(self, image):
        return draw_random_polygon(image)


def draw_pose6D(image, pose6D, points3D, intrinsics, thickness):
    """Draws cube in image by projecting points3D with intrinsics and pose6D.

    # Arguments
        image: Array (H, W).
        pose6D: paz.abstract.Pose6D instance.
        intrinsics: Array (3, 3). Camera intrinsics for projecting
            3D rays into 2D image.
        points3D: Array (num_points, 3).
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    quaternion, translation = pose6D.quaternion, pose6D.translation
    rotation = quaternion_to_rotation_matrix(quaternion)
    points2D = project_to_image(rotation, translation, points3D, intrinsics)
    image = draw_cube(image, points2D.astype(np.int32), thickness=thickness)
    return image


class DrawPoses6D(Processor):
    """Draws multiple cubes in image by projecting points3D.

    # Arguments
        object_sizes: Array (3) indicating (x, y, z) sizes of object.
        camera_intrinsics: Array (3, 3).
            Camera intrinsics for projecting 3D rays into 2D image.
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    def __init__(self, object_sizes, camera_intrinsics, thickness=2):
        self.points3D = build_cube_points3D(*object_sizes)
        self.intrinsics = camera_intrinsics
        self.thickness = thickness

    def call(self, image, poses6D):
        if poses6D is None:
            return image
        if not isinstance(poses6D, list):
            raise ValueError('Poses6D must be a list of Pose6D messages')
        for pose6D in poses6D:
            image = draw_pose6D(
                image, pose6D, self.points3D, self.intrinsics, self.thickness)
        return image


class DrawPose6D(Processor):
    """Draws a single cube in image by projecting points3D.

    # Arguments
        object_sizes: Array (3) indicating (x, y, z) sizes of object.
        camera_intrinsics: Array (3, 3).
            Camera intrinsics for projecting 3D rays into 2D image.
        thickness: Positive integer indicating line thickness.

    # Returns
        Image array (H, W) with drawn inferences.
    """
    def __init__(self, object_sizes, camera_intrinsics, thickness=2):
        self.points3D = build_cube_points3D(*object_sizes)
        self.intrinsics = camera_intrinsics
        self.thickness = thickness

    def call(self, image, pose6D):
        if pose6D is None:
            return image
        image = draw_pose6D(
            image, pose6D, self.points3D, self.intrinsics, self.thickness)
        return image


class DrawHumanSkeleton(Processor):
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
    def __init__(self, dataset, check_scores, link_width=2, keypoint_radius=4):
        super(DrawHumanSkeleton, self).__init__()
        self.link_orders = HUMAN_JOINT_CONFIG[dataset]['part_orders']
        self.link_colors = HUMAN_JOINT_CONFIG[dataset]['part_color']
        self.link_args = HUMAN_JOINT_CONFIG[dataset]['part_arg']
        self.keypoint_colors = HUMAN_JOINT_CONFIG[dataset]['joint_color']
        self.check_scores = check_scores
        self.link_width = link_width
        self.keypoint_radius = keypoint_radius

    def call(self, image, grouped_joints):
        for one_person_joints in grouped_joints:
            image = draw_keypoints_link(
                image, one_person_joints, self.link_args, self.link_orders,
                self.link_colors, self.check_scores, self.link_width)
            image = draw_keypoints(image, one_person_joints,
                                   self.keypoint_colors, self.check_scores,
                                   self.keypoint_radius)
        return image


class DrawHandSkeleton(Processor):
    """ Draw hand pose skeleton on image.

    # Arguments
        image: Array (H, W, 3)
        keypoints: Array. All the joint locations detected by model
                        in the image.
    # Returns
        A numpy array containing pose skeleton.
    """
    def __init__(self, check_scores=False, link_width=2, keypoint_radius=4):
        super(DrawHandSkeleton, self).__init__()
        self.link_orders = MINIMAL_HAND_CONFIG['part_orders']
        self.link_colors = MINIMAL_HAND_CONFIG['part_color']
        self.link_args = MINIMAL_HAND_CONFIG['part_arg']
        self.keypoint_colors = MINIMAL_HAND_CONFIG['joint_color']
        self.check_scores = check_scores
        self.link_width = link_width
        self.keypoint_radius = keypoint_radius

    def call(self, image, keypoints):
        image = draw_keypoints_link(
            image, keypoints, self.link_args, self.link_orders,
            self.link_colors, self.check_scores, self.link_width)
        image = draw_keypoints(image, keypoints, self.keypoint_colors,
                               self.check_scores, self.keypoint_radius)
        return image


class DrawRGBMask(Processor):
    """Draws RGB mask by transforming points3D to RGB space and putting in
        them in their 2D coordinates (points2D)

    # Arguments
        object_sizes: Array (x_size, y_size, z_size)
    """
    def __init__(self, object_sizes):
        super(DrawRGBMask, self).__init__()
        self.object_sizes = object_sizes

    def call(self, image, points2D, points3D):
        image = draw_RGB_mask(image, points2D, points3D, self.object_sizes)
        return image


class DrawRGBMasks(Processor):
    """Draws RGB masks by transforming points3D to RGB space and putting in
        them in their 2D coordinates (points2D)

    # Arguments
        object_sizes: Array (x_size, y_size, z_size)
    """
    def __init__(self, object_sizes):
        super(DrawRGBMasks, self).__init__()
        self.object_sizes = object_sizes

    def call(self, image, points2D, points3D):
        return draw_RGB_masks(image, points2D, points3D, self.object_sizes)


class DrawText(Processor):
    """Draws text to image.

    # Arguments
        color: List. Color of text to
        thickness: Int. Thickness of text.
        scale: Int. Size scale for text.
        message: Str. Text to be added on the image.
        location: List/tuple of int. Pixel corordinte in image to add text.
    """
    def __init__(self, color=GREEN, thickness=2, scale=1):
        super(DrawText, self).__init__()
        self.color = color
        self.thickness = thickness
        self.scale = scale

    def call(self, image, message, location=(50, 50)):
        image = put_text(image, message, location, self.scale,
                         self.color, self.thickness)
        return image
