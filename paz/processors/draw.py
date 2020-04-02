from ..core import ops
from ..core import Processor
import numpy as np


class DrawBoxes2D(Processor):
    """Draws bounding boxes from Boxes2D messages
    # Arguments
        class_names: List of strings.
    """
    def __init__(self, class_names=None, colors=None):
        self.class_names = class_names
        self.colors = colors
        if self.colors is None:
            self.colors = ops.lincolor(len(self.class_names))
        if class_names is not None:
            self.num_classes = len(self.class_names)
            self.class_to_color = dict(zip(self.class_names, self.colors))
        else:
            self.class_to_color = {None: self.colors, '': self.colors}
        super(DrawBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        for box2D in boxes2D:
            x_min, y_min, x_max, y_max = box2D.coordinates
            class_name = box2D.class_name
            color = self.class_to_color[class_name]
            text = '{:0.2f}, {}'.format(box2D.score, class_name)
            ops.put_text(image, text, (x_min, y_min - 10), .7, color, 1)
            ops.draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        return image


class DrawKeypoints2D(Processor):
    """Draws keypoints into image.
    #Arguments
        num_keypoints: Int. Used initialize colors for each keypoint
        radius: Float. Approximate radius of the circle in pixel coordinates.
    """
    def __init__(self, num_keypoints, radius=3, normalized=True):
        super(DrawKeypoints2D, self).__init__()
        self.colors = ops.lincolor(num_keypoints, normalized=normalized)
        self.radius = radius

    def call(self, image, keypoints):
        for keypoint_arg, keypoint in enumerate(keypoints):
            color = self.colors[keypoint_arg]
            ops.draw_circle(image, keypoint.astype('int'), color, self.radius)
        return image


class MakeMosaic(Processor):
    """Draws multiple images as a single mosaic image
    """
    def __init__(self, shape, input_topic='image', output_topic='mosaic'):
        super(MakeMosaic, self).__init__()
        self.shape = shape
        self.input_topic = input_topic
        self.output_topic = output_topic

    def call(self, image):
        mosaic = ops.make_mosaic(image, self.shape)
        return mosaic


class DrawBoxes3D(Processor):
    def __init__(self, camera, class_to_dimensions):
        """Draw boxes 3D of multiple objects
        # Arguments
            camera_intrinsics:
            distortion:
            class_to_dimensions: Dictionary that has as keys the
            class names and as value a list [model_height, model_width]
        """
        # model_height=.1, model_width=0.08):
        super(DrawBoxes3D, self).__init__()
        self.camera = camera
        self.class_to_dimensions = class_to_dimensions
        self.class_to_points = self._make_points(self.class_to_dimensions)

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
        points2D = ops.project_points3D(*args).astype(np.int32)
        ops.draw_cube(image, points2D, thickness=1)
        return image
