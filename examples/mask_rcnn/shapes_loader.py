from paz.abstract.loader import Loader
import numpy as np
from paz.backend.image.draw import draw_circle
from paz.backend.boxes import apply_non_max_suppression
import cv2

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
FILLED = cv2.FILLED


def draw_square(image, center, color, size):
    """Draw a square in an image

    # Arguments
        image: Array ``(H, W, 3)``
        center: List ``(2)`` with ``(x, y)`` values in openCV coordinates.
        size: Float. Length of square size.
        color: List ``(3)`` indicating RGB colors.

    # Returns
        Array ``(H, W, 3)`` with square.
    """
    center_x, center_y = center
    x_min, y_min = center_x - size, center_y - size
    x_max, y_max = center_x + size, center_y + size
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), tuple(color), FILLED)
    return image


def draw_triangle(image, center, color, size):
    """Draw a triangle in an image

    # Arguments
        image: Array ``(H, W, 3)``
        center: List ``(2)`` containing ``(x_center, y_center)``.
        size: Float. Length of square size.
        color: Tuple ``(3)`` indicating the RGB colors.

    # Returns
        Array ``(H, W, 3)`` with triangle.
    """
    center_x, center_y = center
    vertex_A = (center_x, center_y - size)
    vertex_B = (center_x - size, center_y + size)
    vertex_C = (center_x + size, center_y + size)
    points = np.array([[vertex_A, vertex_B, vertex_C]], dtype=np.int32)
    cv2.fillPoly(image, points, tuple(color))
    return image


class Shapes(Loader):
    """ Loader for shapes synthetic dataset.

    # Arguments
        num_samples: Int indicating number of samples to load.
        image_size: (height, width) of input image to load.
        split: String determining the data split to load.
            e.g. `train`, `val` or `test`
        class_names: List of strings or `all`.
        iou_thresh: Float intersection over union.
        max_num_shapes: Int. maximum number of shapes in the image.

    # Returns
        List of dictionaries with keys `image`, `mask`, `box_data`
            containing
    """
    def __init__(self, num_samples, image_size, split='train',
                 class_names='all', iou_thresh=0.3, max_num_shapes=3):
        if class_names == 'all':
            class_names = ['background', 'square', 'circle', 'triangle']
        self.name_to_arg = dict(zip(class_names, range(len(class_names))))
        self.arg_to_name = dict(zip(range(len(class_names)), class_names))
        self.num_samples, self.image_size = num_samples, image_size
        self.labels = ['image', 'masks', 'box_data']
        self.iou_thresh = iou_thresh
        self.max_num_shapes = max_num_shapes
        super(Shapes, self).__init__(None, split, class_names, 'Shapes')

    def load_data(self):
        return [self.load_sample() for arg in range(self.num_samples)]

    def load_sample(self):
        shapes = self._sample_shapes(self.max_num_shapes, *self.image_size)
        boxes = self._compute_bounding_boxes(shapes)
        shapes, boxes = self._filter_shapes(boxes, shapes, self.iou_thresh)
        image = self._draw_shapes(shapes)
        masks = self._draw_masks(shapes)
        class_args = [self.name_to_arg[name[0]] for name in shapes]
        class_args = np.asarray(class_args).reshape(-1, 1)
        box_data = np.concatenate([boxes, class_args], axis=1)
        sample = dict(zip(self.labels, [image, masks, box_data]))
        return sample

    def _sample_shape(self, H, W, offset=20):
        shape = np.random.choice(self.class_names[1:])
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        center_x = np.random.randint(offset, W - offset - 1)
        center_y = np.random.randint(offset, H - offset - 1)
        size = np.random.randint(offset, H // 4)
        return shape, color, (center_x, center_y, size)

    def _sample_shapes(self, num_shapes, H, W, offset=20):
        shapes = []
        for shape_arg in range(num_shapes):
            shapes.append(self._sample_shape(H, W, offset=20))
        return shapes

    def _compute_bounding_box(self, center_x, center_y, size):
        x_min, y_min = center_x - size, center_y - size
        x_max, y_max = center_x + size, center_y + size
        box = [x_min, y_min, x_max, y_max]
        return box

    def _compute_bounding_boxes(self, shapes):
        boxes = []
        for shape in shapes:
            center_x, center_y, size = shape[2]
            box = self._compute_bounding_box(center_x, center_y, size)
            boxes.append(box)
        return np.asarray(boxes)

    def _filter_shapes(self, boxes, shapes, iou_thresh):
        scores = np.ones(len(boxes))  # all shapes have the same score
        args, num_boxes = apply_non_max_suppression(boxes, scores, iou_thresh)
        box_args = args[:num_boxes]
        selected_shapes = []
        for box_arg in box_args:
            selected_shapes.append(shapes[box_arg])
        return selected_shapes, boxes[box_args]

    def _draw_shapes(self, shapes):
        H, W = self.image_size
        background_color = np.random.randint(0, 255, size=3)
        image = np.ones([H, W, 3], dtype=np.uint8)
        image = image * background_color.astype(np.uint8)
        for shape, color, dimensions in shapes:
            image = self._draw_shape(image, shape, dimensions, color)
        return image

    def _draw_shape(self, image, shape, dimensions, color):
        center_x, center_y, size = dimensions
        functions = [draw_square, draw_circle, draw_triangle]
        draw = dict(zip(self.class_names[1:], functions))
        image = draw[shape](image, (center_x, center_y), color, size)
        return image

    def _draw_masks(self, shapes):
        H, W = self.image_size
        class_masks = []
        for class_mask in range(self.num_classes):
            class_masks.append(np.zeros([H, W, 1]))
        class_masks[0] = np.logical_not(class_masks[0])
        for shape_arg, (shape, color, dimensions) in enumerate(shapes):
            mask_arg = self.name_to_arg[shape]
            class_mask = class_masks[mask_arg]
            class_mask = self._draw_shape(
                class_mask, shape, dimensions, (1, 1, 1))
            class_masks[mask_arg] = class_mask
            negative_mask = np.logical_not(class_mask)
            background_mask = class_masks[0].copy()
            class_masks[0] = np.logical_and(negative_mask, background_mask)
        masks = np.concatenate(class_masks, axis=-1).astype(np.uint8)
        return masks

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.image_info = []
        self.image_ids = []
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)