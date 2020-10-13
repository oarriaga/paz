import numpy as np

from ..abstract import Processor
from ..backend.image import lincolor
from ..backend.image import draw_rectangle
from ..backend.image import put_text
from ..backend.image import draw_circle
from ..backend.image import draw_cube
from ..backend.image import draw_random_polygon
from ..backend.keypoints import project_points3D


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
                 weighted=False, scale=0.7):
        self.class_names = class_names
        self.colors = colors
        self.weighted = weighted
        self.scale = scale
        if self.colors is None:
            self.colors = lincolor(len(self.class_names))

        if class_names is not None:
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
            text = '{:0.2f}, {}'.format(box2D.score, class_name)
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
            draw_circle(image, keypoint.astype('int'), color, self.radius)
        return image


class DrawBoxes3D(Processor):
    def __init__(self, camera, class_to_dimensions, thickness=1):
        """Draw boxes 3D of multiple objects

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera''.
            class_to_dimensions: Dictionary that has as keys the
                class names and as value a list [model_height, model_width]
            thickness: Int. Thickness of 3D box
        """
        # model_height=.1, model_width=0.08):
        super(DrawBoxes3D, self).__init__()
        self.camera = camera
        self.class_to_dimensions = class_to_dimensions
        self.class_to_points = self._make_points(self.class_to_dimensions)
        self.thickness = thickness

    def _make_points(self, class_to_dimensions):
        class_to_points = {}
        for class_name, dimensions in self.class_to_dimensions.items():
            height, width = dimensions
            point_1 = [+width, -height, +width]
            point_2 = [+width, -height, -width]
            point_3 = [-width, -height, -width]
            point_4 = [-width, -height, +width]
            point_5 = [+width, +height, +width]
            point_6 = [+width, +height, -width]
            point_7 = [-width, +height, -width]
            point_8 = [-width, +height, +width]
            points = [point_1, point_2, point_3, point_4,
                      point_5, point_6, point_7, point_8]
            class_to_points[class_name] = np.array(points)
        return class_to_points

    def call(self, image, pose6D):
        points3D = self.class_to_points[pose6D.class_name]
        args = (points3D, pose6D, self.camera)
        points2D = project_points3D(*args).astype(np.int32)
        draw_cube(image, points2D, thickness=self.thickness)
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
