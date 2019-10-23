from ..core import ops
from ..core import Processor


class DrawBoxes2D(Processor):
    """Draws bounding boxes from Boxes2D messages
    # Arguments
        class_names: List of strings.
    """
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.colors = ops.lincolor(self.num_classes)
        self.class_to_color = dict(zip(self.class_names, self.colors))
        super(DrawBoxes2D, self).__init__()

    def call(self, kwargs):
        image, boxes2D = kwargs['image'], kwargs['boxes2D']
        for box2D in boxes2D:
            class_name = box2D.class_name
            text = '{:0.2f}, {}'.format(box2D.score, box2D.class_name)
            x_min, y_min, x_max, y_max = box2D.coordinates
            color = self.class_to_color[class_name]
            ops.put_text(image, text, (x_min, y_min - 10), .7, color, 1)
            ops.draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        return kwargs


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

    def call(self, kwargs):
        image, keypoints = kwargs['image'], kwargs['keypoints']
        for keypoint_arg, keypoint in enumerate(keypoints):
            color = self.colors[keypoint_arg]
            ops.draw_circle(image, keypoint.astype('int'), color, self.radius)
        return kwargs
